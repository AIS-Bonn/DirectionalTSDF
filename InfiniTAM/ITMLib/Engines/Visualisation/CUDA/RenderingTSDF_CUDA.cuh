//
// Created by Malte Splietker on 16.03.21.
//

#pragma once

#include "RenderingTSDF_CUDA.h"

#include <ITMLib/Objects/Scene/ITMRepresentationAccess.h>
#include <ITMLib/Utils/ITMGeometry.h>
#include <ITMLib/Utils/ITMCUDAUtils.h>
#include <stdgpu/unordered_map.cuh>

namespace ITMLib
{
__device__ inline ITMVoxel readVoxel(bool& found, const RenderingTSDF tsdf, const Vector3i& voxelIdx)
{
	Vector3i blockPos;
	unsigned short linearIdx;
	voxelToBlockPosAndOffset(voxelIdx, blockPos, linearIdx);

	auto it = tsdf.find(blockPos.toShort());
	if (it == tsdf.end())
	{
		found = false;
		return ITMVoxel();
	}
	found = true;
	return it->second[linearIdx];
}

__device__ inline float readFromSDF_float_uninterpolated(bool& found, const RenderingTSDF tsdf, Vector3f point)
{
	ITMVoxel res = readVoxel(found, tsdf, Vector3i((int) ROUND(point.x), (int) ROUND(point.y), (int) ROUND(point.z)));
	return res.sdf;
}

__device__ inline float readFromSDF_float_interpolated(bool& found, const RenderingTSDF tsdf, Vector3f point)
{
	float res1, res2, v1, v2;
	Vector3f coeff;
	Vector3i pos;
	TO_INT_FLOOR3(pos, coeff, point);
	ITMVoxel voxel;

	found = false;
	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 0));
	v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 0));
	v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res1 = (found, 1.0f - coeff.x) * v1 + coeff.x * v2;

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 0));
	v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 0));
	v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res1 = (found, 1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 1));
	v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 1));
	v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res2 = (found, 1.0f - coeff.x) * v1 + coeff.x * v2;

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 1));
	v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 1));
	v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	found = true;
	return ITMVoxel::valueToFloat((1.0f - coeff.z) * res1 + coeff.z * res2);
}

__device__ inline float
readWithConfidenceFromSDF_float_uninterpolated(float& confidence, bool& found, const RenderingTSDF tsdf, Vector3f point,
                                               const int maxW)
{
	ITMVoxel res = readVoxel(found, tsdf, Vector3i((int) ROUND(point.x), (int) ROUND(point.y), (int) ROUND(point.z)));
	if (found)
		confidence = ITMVoxel::weightToFloat(res.w_depth, maxW);
	else
		confidence = 0;

	return ITMVoxel::valueToFloat(res.sdf);
}

__device__ inline float
readWithConfidenceFromSDF_float_interpolated(bool found, float& confidence, const RenderingTSDF tsdf, Vector3f point,
                                             const int maxW)
{
	float res1, res2, v1, v2;
	float res1_c, res2_c, v1_c, v2_c;
	ITMVoxel voxel;

	Vector3f coeff;
	Vector3i pos;
	TO_INT_FLOOR3(pos, coeff, point);

	found = false;
	confidence = 0;
	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 0));
	v1 = voxel.sdf;
	v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 0));
	v2 = voxel.sdf;
	v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;
	res1_c = (1.0f - coeff.x) * v1_c + coeff.x * v2_c;

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 0));
	v1 = voxel.sdf;
	v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 0));
	v2 = voxel.sdf;
	v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
	res1_c = (1.0f - coeff.y) * res1_c + coeff.y * ((1.0f - coeff.x) * v1_c + coeff.x * v2_c);

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 1));
	v1 = voxel.sdf;
	v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 1));
	v2 = voxel.sdf;
	v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;
	res2_c = (1.0f - coeff.x) * v1_c + coeff.x * v2_c;

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 1));
	v1 = voxel.sdf;
	v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 1));
	v2 = voxel.sdf;
	v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
	res2_c = (1.0f - coeff.y) * res2_c + coeff.y * ((1.0f - coeff.x) * v1_c + coeff.x * v2_c);

	found = true;

	confidence = ITMVoxel::weightToFloat((1.0f - coeff.z) * res1_c + coeff.z * res2_c, maxW);

	return ITMVoxel::valueToFloat((1.0f - coeff.z) * res1 + coeff.z * res2);
}

}
