// Copyright 2019 Malte Splietker

#pragma once

#include "../../Utils/ITMMath.h"

namespace ITMLib
{

using ORUtils::dot;

_CPU_AND_GPU_CODE_
const static size_t N_DIRECTIONS = 6;

_CPU_AND_GPU_CODE_
const static float direction_angle_threshold = 1.1f * float(M_PI_4);

enum class TSDFDirection : std::uint8_t
{
	X_POS = 0,
	X_NEG,
	Y_POS,
	Y_NEG,
	Z_POS,
	Z_NEG,
	NONE = 255
};
typedef std::underlying_type<TSDFDirection>::type TSDFDirection_type;

struct TSDFDirectionColor_
{
	_CPU_AND_GPU_CODE_
	Vector3f operator[](size_t i) const
	{
		const Vector3f directionColor[6] = {
			Vector3f(1, 0, 0),
			Vector3f(0, 1, 0),
			Vector3f(1, 1, 0),
			Vector3f(0, 0, 1),
			Vector3f(1, 0, 1),
			Vector3f(0, 1, 1)
		};
		return directionColor[i];
	}
};

_CPU_AND_GPU_CODE_
const static TSDFDirectionColor_ TSDFDirectionColor; // instantiate to allow operator usage

struct TSDFDirectionVector_
{
	_CPU_AND_GPU_CODE_
	Vector3f operator[](size_t i) const
	{
		const Vector3f directionVectors[N_DIRECTIONS] = {
			Vector3f(1, 0, 0),
			Vector3f(-1, 0, 0),
			Vector3f(0, 1, 0),
			Vector3f(0, -1, 0),
			Vector3f(0, 0, 1),
			Vector3f(0, 0, -1)
		};
		return directionVectors[i];
	}
};

_CPU_AND_GPU_CODE_
const static TSDFDirectionVector_ TSDFDirectionVector;

_CPU_AND_GPU_CODE_
inline const char* TSDFDirectionToString(TSDFDirection direction)
{
	switch (direction)
	{
		case TSDFDirection::X_POS:
			return "X_POS";
		case TSDFDirection::X_NEG:
			return "X_NEG";
		case TSDFDirection::Y_POS:
			return "Y_POS";
		case TSDFDirection::Y_NEG:
			return "Y_NEG";
		case TSDFDirection::Z_POS:
			return "Z_POS";
		case TSDFDirection::Z_NEG:
			return "Z_NEG";
		case TSDFDirection::NONE:
			return "NONE";
	}
	return "UNKNOWN";
}

/**
 * Computes the angle between the normal and direction vector.
 *
 * The angle is computed as acos of the dot product -> [0, pi].
 * @param normal
 * @param angles
 */
_CPU_AND_GPU_CODE_
inline float DirectionAngle(const Vector3f& normal, TSDFDirection direction)
{
	float angleCos = dot(normal, TSDFDirectionVector[static_cast<TSDFDirection_type>(direction)]);
	angleCos = MAX(MIN(angleCos, 1), -1);
	return std::acos(angleCos);
}

/**
 * Computes the angle between the normal and every direction vector.
 *
 * The angle is computed as acos of the dot product, so each angle is in [0, pi].
 * @param normal
 * @param angles
 */
_CPU_AND_GPU_CODE_
inline void ComputeDirectionAngle(const Vector3f& normal, float* angles)
{
	for (size_t i = 0; i < 3; i++)
	{
		angles[2 * i] = DirectionAngle(normal, TSDFDirection(2 * i));
		angles[2 * i + 1] = float(M_PI) - angles[2 * i]; // opposite direction -> opposite angle
	}
}

_CPU_AND_GPU_CODE_
inline float DirectionWeight(float angle)
{
	float width = direction_angle_threshold;

	if (width <= float(M_PI_4) + 1e-6)
	{
		return 1 - MIN(angle / width, 1);
	}

	width /= float(M_PI_2);
	angle /= float(M_PI_2);
	return 1 - MIN((MAX(angle, 1 - width) - (1 - width)) / (2 * width - 1), 1);
}

} // namespace ITMLib
