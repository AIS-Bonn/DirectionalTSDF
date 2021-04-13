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

typedef struct RenderTSDFCache {
	Vector3s blockPos;
	ITMVoxel* ptr;
	_CPU_AND_GPU_CODE_ RenderTSDFCache(void) : blockPos(0x7fff), ptr(nullptr) {}
} IndexCache ;

__device__ inline ITMVoxel readVoxel(bool& found, const RenderingTSDF& tsdf, const Vector3i& voxelIdx, RenderTSDFCache* cache=nullptr)
{
	Vector3i blockPos;
	unsigned short linearIdx;
	voxelToBlockPosAndOffset(voxelIdx, blockPos, linearIdx);

	if (cache and cache->blockPos == blockPos)
	{
		return cache->ptr[linearIdx];
	}

	auto it = tsdf.find(blockPos.toShort());
	if (it == tsdf.end())
	{
		found = false;
		return ITMVoxel();
	}

	if (cache)
	{
		cache->blockPos = blockPos.toShort();
		cache->ptr = it->second;
	}

	found = true;
	return it->second[linearIdx];
}

__device__ inline float readFromSDF_float_uninterpolated(bool& found, const RenderingTSDF& tsdf, Vector3f point)
{
	ITMVoxel res = readVoxel(found, tsdf, Vector3i((int) ROUND(point.x), (int) ROUND(point.y), (int) ROUND(point.z)));
	return res.sdf;
}

__device__ inline float readFromSDF_float_interpolated(bool& found, const RenderingTSDF& tsdf, Vector3f point)
{
	float res1, res2, v1, v2;
	RenderTSDFCache cache;
	Vector3f coeff;
	Vector3i pos;
	TO_INT_FLOOR3(pos, coeff, point);
	ITMVoxel voxel;

	found = false;
	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 0), nullptr);
	v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 0), nullptr);
	v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 0), nullptr);
	v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 0), nullptr);
	v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 1), nullptr);
	v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 1), nullptr);
	v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 1), nullptr);
	v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 1), nullptr);
	v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	found = true;
	return ITMVoxel::valueToFloat((1.0f - coeff.z) * res1 + coeff.z * res2);
}

__device__ inline float
readWithConfidenceFromSDF_float_uninterpolated(float& confidence, bool& found, const RenderingTSDF& tsdf, Vector3f point,
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
readWithConfidenceFromSDF_float_interpolated(bool found, float& confidence, const RenderingTSDF& tsdf, Vector3f point,
                                             const int maxW)
{
	float res1, res2, v1, v2;
	float res1_c, res2_c, v1_c, v2_c;
	ITMVoxel voxel;

	RenderTSDFCache cache;
	Vector3f coeff;
	Vector3i pos;
	TO_INT_FLOOR3(pos, coeff, point);

	found = false;
	confidence = 0;
	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 0), nullptr);
	v1 = voxel.sdf;
	v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 0), nullptr);
	v2 = voxel.sdf;
	v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;
	res1_c = (1.0f - coeff.x) * v1_c + coeff.x * v2_c;

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 0), nullptr);
	v1 = voxel.sdf;
	v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 0), nullptr);
	v2 = voxel.sdf;
	v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
	res1_c = (1.0f - coeff.y) * res1_c + coeff.y * ((1.0f - coeff.x) * v1_c + coeff.x * v2_c);

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 1), nullptr);
	v1 = voxel.sdf;
	v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 1), nullptr);
	v2 = voxel.sdf;
	v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;
	res2_c = (1.0f - coeff.x) * v1_c + coeff.x * v2_c;

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 1), nullptr);
	v1 = voxel.sdf;
	v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 1), nullptr);
	v2 = voxel.sdf;
	v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
	res2_c = (1.0f - coeff.y) * res2_c + coeff.y * ((1.0f - coeff.x) * v1_c + coeff.x * v2_c);

	found = true;

	confidence = ITMVoxel::weightToFloat((1.0f - coeff.z) * res1_c + coeff.z * res2_c, maxW);

	return ITMVoxel::valueToFloat((1.0f - coeff.z) * res1 + coeff.z * res2);
}

#define ReadVoxelSDFDiscardZero(dst, pos) voxel = readVoxel(found, tsdf, pos, nullptr); if (voxel.w_depth == 0) return Vector3f(0, 0, 0); dst = voxel.sdf;
/**
 * Computes raw (unnormalized) gradient from neighboring voxels
 * @tparam TVoxel
 * @tparam TIndex
 * @param voxelData
 * @param voxelIndex
 * @param point
 * @param direction
 * @return
 */
__device__ inline Vector3f computeGradientFromSDF(const RenderingTSDF& tsdf, const Vector3f& point)
{
	bool found;

	RenderTSDFCache cache;
	ITMVoxel voxel;

	Vector3f ret(0, 0, 0);
	Vector3f coeff; Vector3i pos; TO_INT_FLOOR3(pos, coeff, point);
	Vector3f ncoeff(1.0f - coeff.x, 1.0f - coeff.y, 1.0f - coeff.z);

	// all 8 values are going to be reused several times
	Vector4f front, back;
	ReadVoxelSDFDiscardZero(front.x, pos + Vector3i(0, 0, 0));
	ReadVoxelSDFDiscardZero(front.y, pos + Vector3i(1, 0, 0));
	ReadVoxelSDFDiscardZero(front.z, pos + Vector3i(0, 1, 0));
	ReadVoxelSDFDiscardZero(front.w, pos + Vector3i(1, 1, 0));
	ReadVoxelSDFDiscardZero(back.x, pos + Vector3i(0, 0, 1));
	ReadVoxelSDFDiscardZero(back.y, pos + Vector3i(1, 0, 1));
	ReadVoxelSDFDiscardZero(back.z, pos + Vector3i(0, 1, 1));
	ReadVoxelSDFDiscardZero(back.w, pos + Vector3i(1, 1, 1));

	Vector4f tmp;
	float p1, p2, v1;
	// gradient x
	p1 = front.x * ncoeff.y * ncoeff.z +
	     front.z *  coeff.y * ncoeff.z +
	     back.x  * ncoeff.y *  coeff.z +
	     back.z  *  coeff.y *  coeff.z;
	ReadVoxelSDFDiscardZero(tmp.x, pos + Vector3i(-1, 0, 0));
	ReadVoxelSDFDiscardZero(tmp.y, pos + Vector3i(-1, 1, 0));
	ReadVoxelSDFDiscardZero(tmp.z, pos + Vector3i(-1, 0, 1));
	ReadVoxelSDFDiscardZero(tmp.w, pos + Vector3i(-1, 1, 1));
	p2 = tmp.x * ncoeff.y * ncoeff.z +
	     tmp.y *  coeff.y * ncoeff.z +
	     tmp.z * ncoeff.y *  coeff.z +
	     tmp.w *  coeff.y *  coeff.z;
	v1 = p1 * coeff.x + p2 * ncoeff.x;

	p1 = front.y * ncoeff.y * ncoeff.z +
	     front.w *  coeff.y * ncoeff.z +
	     back.y  * ncoeff.y *  coeff.z +
	     back.w  *  coeff.y *  coeff.z;
	ReadVoxelSDFDiscardZero(tmp.x, pos + Vector3i(2, 0, 0));
	ReadVoxelSDFDiscardZero(tmp.y, pos + Vector3i(2, 1, 0));
	ReadVoxelSDFDiscardZero(tmp.z, pos + Vector3i(2, 0, 1));
	ReadVoxelSDFDiscardZero(tmp.w, pos + Vector3i(2, 1, 1));
	p2 = tmp.x * ncoeff.y * ncoeff.z +
	     tmp.y *  coeff.y * ncoeff.z +
	     tmp.z * ncoeff.y *  coeff.z +
	     tmp.w *  coeff.y *  coeff.z;

	ret.x = ITMVoxel::valueToFloat(p1 * ncoeff.x + p2 * coeff.x - v1);

	// gradient y
	p1 = front.x * ncoeff.x * ncoeff.z +
	     front.y *  coeff.x * ncoeff.z +
	     back.x  * ncoeff.x *  coeff.z +
	     back.y  *  coeff.x *  coeff.z;
	ReadVoxelSDFDiscardZero(tmp.x, pos + Vector3i(0, -1, 0));
	ReadVoxelSDFDiscardZero(tmp.y, pos + Vector3i(1, -1, 0));
	ReadVoxelSDFDiscardZero(tmp.z, pos + Vector3i(0, -1, 1));
	ReadVoxelSDFDiscardZero(tmp.w, pos + Vector3i(1, -1, 1));
	p2 = tmp.x * ncoeff.x * ncoeff.z +
	     tmp.y *  coeff.x * ncoeff.z +
	     tmp.z * ncoeff.x *  coeff.z +
	     tmp.w *  coeff.x *  coeff.z;
	v1 = p1 * coeff.y + p2 * ncoeff.y;

	p1 = front.z * ncoeff.x * ncoeff.z +
	     front.w *  coeff.x * ncoeff.z +
	     back.z  * ncoeff.x *  coeff.z +
	     back.w  *  coeff.x *  coeff.z;
	ReadVoxelSDFDiscardZero(tmp.x, pos + Vector3i(0, 2, 0));
	ReadVoxelSDFDiscardZero(tmp.y, pos + Vector3i(1, 2, 0));
	ReadVoxelSDFDiscardZero(tmp.z, pos + Vector3i(0, 2, 1));
	ReadVoxelSDFDiscardZero(tmp.w, pos + Vector3i(1, 2, 1));
	p2 = tmp.x * ncoeff.x * ncoeff.z +
	     tmp.y *  coeff.x * ncoeff.z +
	     tmp.z * ncoeff.x *  coeff.z +
	     tmp.w *  coeff.x *  coeff.z;

	ret.y = ITMVoxel::valueToFloat(p1 * ncoeff.y + p2 * coeff.y - v1);

	// gradient z
	p1 = front.x * ncoeff.x * ncoeff.y +
	     front.y *  coeff.x * ncoeff.y +
	     front.z * ncoeff.x *  coeff.y +
	     front.w *  coeff.x *  coeff.y;
	ReadVoxelSDFDiscardZero(tmp.x, pos + Vector3i(0, 0, -1));
	ReadVoxelSDFDiscardZero(tmp.y, pos + Vector3i(1, 0, -1));
	ReadVoxelSDFDiscardZero(tmp.z, pos + Vector3i(0, 1, -1));
	ReadVoxelSDFDiscardZero(tmp.w, pos + Vector3i(1, 1, -1));
	p2 = tmp.x * ncoeff.x * ncoeff.y +
	     tmp.y *  coeff.x * ncoeff.y +
	     tmp.z * ncoeff.x *  coeff.y +
	     tmp.w *  coeff.x *  coeff.y;
	v1 = p1 * coeff.z + p2 * ncoeff.z;

	p1 = back.x * ncoeff.x * ncoeff.y +
	     back.y *  coeff.x * ncoeff.y +
	     back.z * ncoeff.x *  coeff.y +
	     back.w *  coeff.x *  coeff.y;
	ReadVoxelSDFDiscardZero(tmp.x, pos + Vector3i(0, 0, 2));
	ReadVoxelSDFDiscardZero(tmp.y, pos + Vector3i(1, 0, 2));
	ReadVoxelSDFDiscardZero(tmp.z, pos + Vector3i(0, 1, 2));
	ReadVoxelSDFDiscardZero(tmp.w, pos + Vector3i(1, 1, 2));
	p2 = tmp.x * ncoeff.x * ncoeff.y +
	     tmp.y *  coeff.x * ncoeff.y +
	     tmp.z * ncoeff.x *  coeff.y +
	     tmp.w *  coeff.x *  coeff.y;

	ret.z = ITMVoxel::valueToFloat(p1 * ncoeff.z + p2 * coeff.z - v1);
	return ret;
}

/**
 * @tparam filter
 * @tparam TVoxel
 * @tparam TIndex
 * @param voxelData
 * @param voxelIndex
 * @param point
 * @param direction
 * @param tau = voxelSize / truncationDistance (normalized value per 1 voxel). If 0 don't filter
 * @return
 */
__device__ inline Vector3f computeSingleNormalFromSDF(const RenderingTSDF& tsdf, const Vector3f &point,
                                                              const float tau=0.0)
{
	Vector3f gradient = computeGradientFromSDF(tsdf, point);

	if (gradient == Vector3f(0, 0, 0))
		return gradient;

	if (tau > 0)
	{
		// Check each direction maximum 2 * truncationDistance / voxelSize + margin
		if (abs(gradient.x) > 2.4 * tau or abs(gradient.y) > 2.4 * tau or abs(gradient.z) > 2.4 * tau) return Vector3f(0, 0, 0);
		// Check, if gradient too unreliable (very close values in neighboring voxels). minimum expected length: (2 * tau)^2
		if (ORUtils::dot(gradient, gradient) < 2 * (tau * tau)) return Vector3f(0, 0, 0);
	}

	return gradient.normalised();
}

__device__ inline Vector4f readFromSDF_color4u_interpolated(const RenderingTSDF& tsdf, const Vector3f & point)
{
	ITMVoxel voxel;
	RenderTSDFCache cache;
	Vector3f color(0.0f); float w_color = 0;
	Vector3f coeff; Vector3i pos; TO_INT_FLOOR3(pos, coeff, point);

	bool found;
	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 0), nullptr);
	color += (1.0f - coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();

	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 0), nullptr);
	color += (coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 0), nullptr);
	color += (1.0f - coeff.x) * (coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();

	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 0), nullptr);
	color += (coeff.x) * (coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 0, 1), nullptr);
	color += (1.0f - coeff.x) * (1.0f - coeff.y) * coeff.z * voxel.clr.toFloat();

	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 0, 1), nullptr);
	color += (coeff.x) * (1.0f - coeff.y) * coeff.z * voxel.clr.toFloat();

	voxel = readVoxel(found, tsdf, pos + Vector3i(0, 1, 1), nullptr);
	color += (1.0f - coeff.x) * (coeff.y) * coeff.z * voxel.clr.toFloat();

	voxel = readVoxel(found, tsdf, pos + Vector3i(1, 1, 1), nullptr);
	color += (coeff.x) * (coeff.y) * coeff.z * voxel.clr.toFloat();

	return Vector4f(color / 255.0f, 255.0f);
}

}