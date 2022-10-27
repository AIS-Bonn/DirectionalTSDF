// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <limits>
#include "../../Utils/ITMMath.h"

namespace ITMLib
{

/** \brief
    Stores the information of a single voxel in the volume
*/
struct alignas(16) ITMVoxel_f_rgb
{
	typedef float WeightType;

	_CPU_AND_GPU_CODE_ static float SDF_initialValue()
	{ return 1.0f; }

	_CPU_AND_GPU_CODE_ static float valueToFloat(float x)
	{ return x; }

	_CPU_AND_GPU_CODE_ static float floatToValue(float x)
	{ return x; }

	_CPU_AND_GPU_CODE_ static float weightToFloat(WeightType w, int maxW)
	{
		return static_cast<float>(w);
	}

	_CPU_AND_GPU_CODE_ static WeightType floatToWeight(float w, int maxW)
	{
		return static_cast<WeightType>(w);
	}

	static const bool hasColorInformation = true;

	/** Value of the truncated signed distance transformation. */
	float sdf;
	/** Number of fused observations that make up @p sdf. */
	WeightType w_depth;
	/** RGB colour information stored for this voxel. */
	Vector3u clr;
	/** Number of observations that made up @p clr. */
	WeightType w_color;

	_CPU_AND_GPU_CODE_ ITMVoxel_f_rgb()
	{
		sdf = SDF_initialValue();
		w_depth = 0;
		clr = Vector3u((uchar) 0);
		w_color = 0;
	}
};

/** \brief
    Stores the information of a single voxel in the volume
*/
struct alignas(16) ITMVoxel_s_rgb
{
	typedef ushort WeightType;

	_CPU_AND_GPU_CODE_ static short SDF_initialValue()
	{ return 32767; }

	_CPU_AND_GPU_CODE_ static float valueToFloat(short x)
	{ return (float) (x) / 32767.0f; }

	_CPU_AND_GPU_CODE_ static short floatToValue(float x)
	{ return (short) ((x) * 32767.0f); }

	_CPU_AND_GPU_CODE_ static float weightToFloat(WeightType w, int maxW)
	{
		return static_cast<float>(w) * maxW / 65535.0f;
	}

	_CPU_AND_GPU_CODE_ static WeightType floatToWeight(float w, int maxW)
	{
		return static_cast<WeightType>(round(w / maxW * 65535.0f));
	}

	static const bool hasColorInformation = true;

	/** Value of the truncated signed distance transformation. */
	short sdf;
	/** Number of fused observations that make up @p sdf. */
	WeightType w_depth;
	/** RGB colour information stored for this voxel. */
	Vector3u clr;
	/** Number of observations that made up @p clr. */
	WeightType w_color;

	_CPU_AND_GPU_CODE_ ITMVoxel_s_rgb()
	{
		sdf = SDF_initialValue();
		w_depth = 0;
		clr = Vector3u((uchar) 0);
		w_color = 0;
	}
};

struct alignas(4) ITMVoxel_s
{
	typedef ushort WeightType;

	_CPU_AND_GPU_CODE_ static short SDF_initialValue()
	{ return 32767; }

	_CPU_AND_GPU_CODE_ static float valueToFloat(float x)
	{ return (float) (x) / 32767.0f; }

	_CPU_AND_GPU_CODE_ static short floatToValue(float x)
	{ return (short) ((x) * 32767.0f); }

	_CPU_AND_GPU_CODE_ static float weightToFloat(WeightType w, int maxW)
	{
		return static_cast<float>(w) * maxW / 65535.0f;
	}

	_CPU_AND_GPU_CODE_ static WeightType floatToWeight(float w, int maxW)
	{
		return static_cast<WeightType>(round(w / maxW * 65535.0f));
	}

	static const bool hasColorInformation = false;

	/** Value of the truncated signed distance transformation. */
	short sdf;
	/** Number of fused observations that make up @p sdf. */
	WeightType w_depth;

	_CPU_AND_GPU_CODE_ ITMVoxel_s()
	{
		sdf = SDF_initialValue();
		w_depth = 0;
	}
};

struct alignas(8) ITMVoxel_f
{
	typedef float WeightType;

	_CPU_AND_GPU_CODE_ static float SDF_initialValue()
	{ return 1.0f; }

	_CPU_AND_GPU_CODE_ static float valueToFloat(float x)
	{ return x; }

	_CPU_AND_GPU_CODE_ static float floatToValue(float x)
	{ return x; }

	_CPU_AND_GPU_CODE_ static float weightToFloat(WeightType w, int maxW)
	{
		return static_cast<float>(w);
	}

	_CPU_AND_GPU_CODE_ static WeightType floatToWeight(float w, int maxW)
	{
		return static_cast<WeightType>(w);
	}

	static const bool hasColorInformation = false;

	/** Value of the truncated signed distance transformation. */
	float sdf;
	/** Number of fused observations that make up @p sdf. */
	WeightType w_depth;

	_CPU_AND_GPU_CODE_ ITMVoxel_f()
	{
		sdf = SDF_initialValue();
		w_depth = 0;
	}
};

} // namespace ITMLib