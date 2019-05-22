// Copyright 2019 Malte Splietker

#pragma once

#include "../../Utils/ITMMath.h"

namespace ITMLib
{

using ORUtils::dot;

_CPU_AND_GPU_CODE_
const static size_t N_DIRECTIONS = 6;

_CPU_AND_GPU_CODE_
const static float direction_weight_threshold = 0.3826834323650898f; // approx of sin(pi/8)

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
 * Computes the compliance of the given normal with each directions.
 *
 * The weight is computed by a dot product, so each weight is in [-1, 1]. Negative values mean non-compliance
 * and a value of 1 is full compliance.
 * @param normal
 * @param weights
 */
_CPU_AND_GPU_CODE_
inline void ComputeDirectionWeights(const Vector3f& normal, float weights[N_DIRECTIONS])
{
	const Vector3f TSDFDirectionVectors[N_DIRECTIONS] = {
		Vector3f(1,  0,  0),
		Vector3f(-1, 0,  0),
		Vector3f(0,  1,  0),
		Vector3f(0,  -1, 0),
		Vector3f(0,  0,  1),
		Vector3f(0,  0,  -1)
	};
	for (size_t i = 0; i < 3; i++)
	{
		weights[2 * i] = dot(normal, TSDFDirectionVectors[2 * i]);
		weights[2 * i + 1] = -weights[2 * i]; // opposite direction -> negative value
	}
}

_CPU_AND_GPU_CODE_
inline float DirectionWeight(const Vector3f& normal, TSDFDirection direction)
{
	const Vector3f TSDFDirectionVectors[N_DIRECTIONS] = {
		Vector3f(1,  0,  0),
		Vector3f(-1, 0,  0),
		Vector3f(0,  1,  0),
		Vector3f(0,  -1, 0),
		Vector3f(0,  0,  1),
		Vector3f(0,  0,  -1)
	};
	return dot(normal, TSDFDirectionVectors[static_cast<TSDFDirection_type>(direction)]);
}

//__device__
//short FilterMCIndexDirection(const short mc_index, const TSDFDirection direction, const float sdf[8]);

///**
// * Check, whether the given MC index is compatible to the direction.
// *
// * @param mc_index MC index
// * @param direction Direction
// * @param sdf SDF values of voxel corners
// * @return
// */
//__device__
//bool IsMCIndexDirectionCompatible(const short mc_index, const TSDFDirection direction, const float sdf[8]);
//}

//__device__
//short FilterMCIndexDirection(const short mc_index, const TSDFDirection direction, const float sdf[8])
//{
//	if (mc_index <= 0 or mc_index == 255)
//		return mc_index;
//
//	short new_index = 0;
//	for (int component = 0; component < 4 and kIndexDecomposition[mc_index][component] != -1; component++)
//	{
//		const short part_idx = kIndexDecomposition[mc_index][component];
//		if (not IsMCIndexDirectionCompatible(part_idx, direction, sdf))
//			continue;
//		new_index |= part_idx;
//	}
//
//	if (new_index == 0)
//	{ // If 0 after filtering -> invalidate, so it doesn't affect other directions during later filtering process
//		new_index = -1;
//	}
//	return new_index;
//}


//__device__
//bool IsMCIndexDirectionCompatible(const short mc_index, const TSDFDirection direction, const float sdf[8])
//{
//	// Table containing for each direction:
//	// 4 opposite edge pairs, each of which is checked individually.
//	const static size_t view_direction_edges_to_check[6][8] = {
//		{0, 4, 1, 5, 2,  6,  3,  7},  // Y_POS
//		{4, 0, 5, 1, 6,  2,  7,  3},  // Y_NEG
//		{1, 3, 5, 7, 9,  8,  10, 11}, // X_POS
//		{3, 1, 7, 5, 8,  9,  11, 10}, // X_NEG
//		{2, 0, 6, 4, 10, 9,  11, 8},  // Z_NEG
//		{0, 2, 4, 6, 8,  11, 9,  10}  // Z_POS
//	};
//	if (kIndexDirectionCompatibility[mc_index][static_cast<size_t>(direction)] == 0)
//		return false;
//	if (kIndexDirectionCompatibility[mc_index][static_cast<size_t>(direction)] == 2)
//	{
//		for (int e = 0; e < 4; e++)
//		{
//			const size_t edge_idx = view_direction_edges_to_check[static_cast<size_t>(direction)][2 * e];
//			const size_t opposite_edge_idx = view_direction_edges_to_check[static_cast<size_t>(direction)][2 * e + 1];
//			int2 edge = kEdgeEndpointVertices[edge_idx];
//			int2 opposite_edge = kEdgeEndpointVertices[opposite_edge_idx];
//
//			int2 endpoint_values;
//			endpoint_values.x = (mc_index & (1 << edge.x)) > 0;
//			endpoint_values.y = (mc_index & (1 << edge.y)) > 0;
//
//			// If edge has NO zero-crossing -> continue
//			if (endpoint_values.x + endpoint_values.y != 1)
//				continue;
//
//			// Swap vertex indices, s.t. first endpoint is behind the surface
//			if (endpoint_values.y == 1)
//			{
//				int tmp;
//				tmp = edge.x;
//				edge.x = edge.y;
//				edge.y = tmp;
//				tmp = opposite_edge.x;
//				opposite_edge.x = opposite_edge.y;
//				opposite_edge.y = tmp;
//			}
//
//			float offset = InterpolateSurfaceOffset(sdf[edge.x], sdf[edge.y], 0);
//			float opposite_offset = InterpolateSurfaceOffset(sdf[opposite_edge.x], sdf[opposite_edge.y], 0);
//
//			// If interpolated surface more than 90 degrees from view direction vector -> discard
//			if (offset > opposite_offset)
//			{
//				return false;
//			}
////      if (fabs(opposite_offset - offset) < 0.5)
////      {
////        return false;
////      }
//		}
//	}
//	return true;
//}

} // namespace ITMLib
