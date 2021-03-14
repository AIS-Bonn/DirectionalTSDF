// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMVoxelBlockHash.h"
#include "ITMDirectional.h"
#include "ITMLib/Utils/ITMGeometry.h"

namespace ITMLib
{

_CPU_AND_GPU_CODE_ inline ITMHashEntry getHashEntry(
	const CONSTPTR(ITMVoxelBlockHash::IndexData) *hashTable, Vector3i blockPos,
	const TSDFDirection direction=TSDFDirection::NONE)
{
	int hashIdx;
	if (direction != TSDFDirection::NONE)
		hashIdx = hashIndex(blockPos, direction);
	else
		hashIdx = hashIndex(blockPos);

	ITMHashEntry hashEntry;
	while (true)
	{
		hashEntry = hashTable[hashIdx];

		if (IS_EQUAL3(hashEntry.pos, blockPos) && hashEntry.ptr >= 0)
		{
			return hashEntry;
		}

		if (hashEntry.offset < 1) break;
		hashIdx = SDF_BUCKET_NUM + hashEntry.offset - 1;
	}

	hashEntry.ptr = -1;
	return hashEntry;
}

_CPU_AND_GPU_CODE_ inline int findVoxel(const CONSTPTR(ITMLib::ITMVoxelBlockHash::IndexData) *voxelIndex,
	const THREADPTR(Vector3i) & point, const TSDFDirection direction,
	THREADPTR(int) &vmIndex, THREADPTR(ITMLib::ITMVoxelBlockHash::IndexCache) & cache)
{
	Vector3i blockPos;
	unsigned short linearIdx;
	voxelToBlockPosAndOffset(point, blockPos, linearIdx);

	if IS_EQUAL3(blockPos, cache.blockPos)
	{
		vmIndex = true;
		return cache.blockPtr + linearIdx;
	}

	ITMHashEntry hashEntry = getHashEntry(voxelIndex, blockPos, direction);
	vmIndex = hashEntry.IsValid();
	if (vmIndex)
	{
		cache.blockPos = blockPos; cache.blockPtr = hashEntry.ptr * SDF_BLOCK_SIZE3;
		return cache.blockPtr + linearIdx;
	}
	return -1;
}

_CPU_AND_GPU_CODE_ inline int findVoxel(const CONSTPTR(ITMLib::ITMVoxelBlockHash::IndexData) *voxelIndex,
                                        const THREADPTR(Vector3i) & point, THREADPTR(int) &vmIndex,
                                        THREADPTR(ITMLib::ITMVoxelBlockHash::IndexCache) & cache)
{
	return findVoxel(voxelIndex, point, TSDFDirection::NONE, vmIndex, cache);
}

template<class TVoxel>
_CPU_AND_GPU_CODE_ inline TVoxel readVoxel(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(ITMLib::ITMVoxelBlockHash::IndexData) *voxelIndex,
	const THREADPTR(Vector3i) & point, const TSDFDirection direction,
	THREADPTR(int) &vmIndex, THREADPTR(ITMLib::ITMVoxelBlockHash::IndexCache) & cache)
{
	Vector3i blockPos;
	unsigned short linearIdx;
	voxelToBlockPosAndOffset(point, blockPos, linearIdx);

	if IS_EQUAL3(blockPos, cache.blockPos)
	{
		vmIndex = true;
		return voxelData[cache.blockPtr + linearIdx];
	}

	ITMHashEntry hashEntry = getHashEntry(voxelIndex, blockPos, direction);
	vmIndex = hashEntry.IsValid();
	if (vmIndex)
	{
		cache.blockPos = blockPos; cache.blockPtr = hashEntry.ptr * SDF_BLOCK_SIZE3;
		return voxelData[cache.blockPtr + linearIdx];
	}
	return TVoxel();
}

template<class TVoxel>
_CPU_AND_GPU_CODE_ inline TVoxel readVoxel(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(ITMLib::ITMVoxelBlockHash::IndexData) *voxelIndex,
	Vector3i point, const TSDFDirection direction, THREADPTR(int) &vmIndex)
{
	ITMLib::ITMVoxelBlockHash::IndexCache cache;
	return readVoxel(voxelData, voxelIndex, point, direction, vmIndex, cache);
}

template<class TVoxel, class TIndex, class TCache>
_CPU_AND_GPU_CODE_ inline float readFromSDF_float_uninterpolated(const CONSTPTR(TVoxel) *voxelData,
	const CONSTPTR(TIndex) *voxelIndex, Vector3f point, const TSDFDirection direction, THREADPTR(int) &vmIndex, THREADPTR(TCache) & cache)
{
	TVoxel res = readVoxel(voxelData, voxelIndex,
		Vector3i((int)ROUND(point.x), (int)ROUND(point.y), (int)ROUND(point.z)), direction, vmIndex, cache);
	return TVoxel::valueToFloat(res.sdf);
}

#define DISCARD_ZERO_WEIGHT if (voxel.w_depth <= 0) return 1;

template<class TVoxel, class TIndex, class TCache>
_CPU_AND_GPU_CODE_ inline float readFromSDF_float_interpolated(const CONSTPTR(TVoxel) *voxelData,
	const CONSTPTR(TIndex) *voxelIndex, Vector3f point, const TSDFDirection direction, THREADPTR(int) &vmIndex, THREADPTR(TCache) & cache)
{
	float res1, res2, v1, v2;
	Vector3f coeff; Vector3i pos; TO_INT_FLOOR3(pos, coeff, point);
	TVoxel voxel;

	vmIndex = false;
	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 0), direction, vmIndex, cache); v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 0), direction, vmIndex, cache); v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 0), direction, vmIndex, cache); v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 0), direction, vmIndex, cache); v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 1), direction, vmIndex, cache); v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 1), direction, vmIndex, cache); v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 1), direction, vmIndex, cache); v1 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 1), direction, vmIndex, cache); v2 = voxel.sdf;
	DISCARD_ZERO_WEIGHT
	res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	vmIndex = true;
	return TVoxel::valueToFloat((1.0f - coeff.z) * res1 + coeff.z * res2);
}

template<class TVoxel, class TIndex, class TCache>
_CPU_AND_GPU_CODE_ inline float readWithConfidenceFromSDF_float_uninterpolated(float &confidence, const TVoxel *voxelData,
                                                                 const TIndex *voxelIndex, Vector3f point, const TSDFDirection direction, const int maxW, int &vmIndex, TCache & cache)
{
	TVoxel res = readVoxel(voxelData, voxelIndex,
	                       Vector3i((int)ROUND(point.x), (int)ROUND(point.y), (int)ROUND(point.z)), direction, vmIndex, cache);
	if (vmIndex)
		confidence = TVoxel::weightToFloat(res.w_depth, maxW);
	else
		confidence = 0;

	return TVoxel::valueToFloat(res.sdf);
}

template<class TVoxel, class TIndex, class TCache>
_CPU_AND_GPU_CODE_ inline float readWithConfidenceFromSDF_float_interpolated(THREADPTR(float) &confidence, const CONSTPTR(TVoxel) *voxelData,
	const CONSTPTR(TIndex) *voxelIndex, Vector3f point, const TSDFDirection direction, const int maxW, THREADPTR(int) &vmIndex, THREADPTR(TCache) & cache)
{
	float res1, res2, v1, v2;
	float res1_c, res2_c, v1_c, v2_c;
	TVoxel voxel;

	Vector3f coeff; Vector3i pos; TO_INT_FLOOR3(pos, coeff, point);

	vmIndex = false;
	confidence = 0;
	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 0), direction, vmIndex, cache); v1 = voxel.sdf; v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 0), direction, vmIndex, cache); v2 = voxel.sdf; v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;
	res1_c = (1.0f - coeff.x) * v1_c + coeff.x * v2_c;

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 0), direction, vmIndex, cache); v1 = voxel.sdf; v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 0), direction, vmIndex, cache); v2 = voxel.sdf; v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
	res1_c = (1.0f - coeff.y) * res1_c + coeff.y * ((1.0f - coeff.x) * v1_c + coeff.x * v2_c);

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 1), direction, vmIndex, cache); v1 = voxel.sdf; v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 1), direction, vmIndex, cache); v2 = voxel.sdf; v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;
	res2_c = (1.0f - coeff.x) * v1_c + coeff.x * v2_c;

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 1), direction, vmIndex, cache); v1 = voxel.sdf; v1_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 1), direction, vmIndex, cache); v2 = voxel.sdf; v2_c = voxel.w_depth;
	DISCARD_ZERO_WEIGHT
	res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);
	res2_c = (1.0f - coeff.y) * res2_c + coeff.y * ((1.0f - coeff.x) * v1_c + coeff.x * v2_c);

	vmIndex = true;

	confidence = TVoxel::weightToFloat((1.0f - coeff.z) * res1_c + coeff.z * res2_c, maxW);

	return TVoxel::valueToFloat((1.0f - coeff.z) * res1 + coeff.z * res2);
}

template<class TVoxel, class TIndex, class TCache>
_CPU_AND_GPU_CODE_ inline float readFromSDF_float_interpolated(const CONSTPTR(TVoxel) *voxelData,
	const CONSTPTR(TIndex) *voxelIndex, Vector3f point, const TSDFDirection direction, THREADPTR(int) &vmIndex, THREADPTR(TCache) & cache, int & maxW)
{
	float res1, res2, v1, v2;
	Vector3f coeff; Vector3i pos; TO_INT_FLOOR3(pos, coeff, point);

	{
		const TVoxel & v = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 0), direction, vmIndex, cache);
		v1 = v.sdf;
		maxW = v.w_depth;
	}
	{
		const TVoxel & v = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 0), direction, vmIndex, cache);
		v2 = v.sdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	res1 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	{
		const TVoxel & v = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 0), direction, vmIndex, cache);
		v1 = v.sdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	{
		const TVoxel & v = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 0), direction, vmIndex, cache);
		v2 = v.sdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	res1 = (1.0f - coeff.y) * res1 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	{
		const TVoxel & v = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 1), direction, vmIndex, cache);
		v1 = v.sdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	{
		const TVoxel & v = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 1), direction, vmIndex, cache);
		v2 = v.sdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	res2 = (1.0f - coeff.x) * v1 + coeff.x * v2;

	{
		const TVoxel & v = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 1), direction, vmIndex, cache);
		v1 = v.sdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	{
		const TVoxel & v = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 1), direction, vmIndex, cache);
		v2 = v.sdf;
		if (v.w_depth > maxW) maxW = v.w_depth;
	}
	res2 = (1.0f - coeff.y) * res2 + coeff.y * ((1.0f - coeff.x) * v1 + coeff.x * v2);

	vmIndex = true;
	return TVoxel::valueToFloat((1.0f - coeff.z) * res1 + coeff.z * res2);
}

template<class TVoxel, class TIndex, class TCache>
_CPU_AND_GPU_CODE_ inline Vector4f readFromSDF_color4u_uninterpolated(float& confidence, const CONSTPTR(TVoxel) *voxelData,
																																			const CONSTPTR(TIndex) *voxelIndex, const THREADPTR(Vector3f) & point, const TSDFDirection direction, THREADPTR(TCache) & cache,
																																			const int maxW)
{
	TVoxel voxel;
	Vector3f color(0.0f);
	Vector3f coeff; Vector3i pos; TO_INT_FLOOR3(pos, coeff, point);

	confidence = 0;

	int vmIndex;
	voxel = readVoxel(voxelData, voxelIndex, pos, direction, vmIndex, cache);
	confidence = TVoxel::weightToFloat(voxel.w_color, maxW);

	return Vector4f(voxel.clr.toFloat() / 255.0f, 255.0f);
}

template<class TVoxel, class TIndex, class TCache>
_CPU_AND_GPU_CODE_ inline Vector4f readFromSDF_color4u_interpolated(float& confidence, const CONSTPTR(TVoxel) *voxelData,
	const CONSTPTR(TIndex) *voxelIndex, const THREADPTR(Vector3f) & point, const TSDFDirection direction, THREADPTR(TCache) & cache,
	const int maxW)
{
	TVoxel voxel;
	Vector3f color(0.0f); float w_color = 0;
	Vector3f coeff; Vector3i pos; TO_INT_FLOOR3(pos, coeff, point);

	confidence = 0;

	int vmIndex;
	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 0), direction, vmIndex, cache);
	color += (1.0f - coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();
	w_color += (1.0f - coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * voxel.w_color;

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 0), direction, vmIndex, cache);
	color += (coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();
	w_color += (coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * voxel.w_color;

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 0), direction, vmIndex, cache);
	color += (1.0f - coeff.x) * (coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();
	w_color += (1.0f - coeff.x) * (coeff.y) * (1.0f - coeff.z) * voxel.w_color;

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 0), direction, vmIndex, cache);
	color += (coeff.x) * (coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();
	w_color += (coeff.x) * (coeff.y) * (1.0f - coeff.z) * voxel.w_color;

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 1), direction, vmIndex, cache);
	color += (1.0f - coeff.x) * (1.0f - coeff.y) * coeff.z * voxel.clr.toFloat();
	w_color += (1.0f - coeff.x) * (1.0f - coeff.y) * coeff.z * voxel.w_color;

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 1), direction, vmIndex, cache);
	color += (coeff.x) * (1.0f - coeff.y) * coeff.z * voxel.clr.toFloat();
	w_color += (coeff.x) * (1.0f - coeff.y) * coeff.z * voxel.w_color;

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 1), direction, vmIndex, cache);
	color += (1.0f - coeff.x) * (coeff.y) * coeff.z * voxel.clr.toFloat();
	w_color += (1.0f - coeff.x) * (coeff.y) * coeff.z * voxel.w_color;

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 1), direction, vmIndex, cache);
	color += (coeff.x) * (coeff.y) * coeff.z * voxel.clr.toFloat();
	w_color += (coeff.x) * (coeff.y) * coeff.z * voxel.w_color;

	confidence = TVoxel::weightToFloat(w_color, maxW);

	return Vector4f(color / 255.0f, 255.0f);
}

template<class TVoxel, class TIndex, class TCache>
_CPU_AND_GPU_CODE_ inline Vector4f readFromSDF_color4u_interpolated(const CONSTPTR(TVoxel) *voxelData,
                                                                    const CONSTPTR(TIndex) *voxelIndex, const THREADPTR(Vector3f) & point, const TSDFDirection direction, THREADPTR(TCache) & cache)
{
	TVoxel voxel;
	Vector3f color(0.0f); float w_color = 0;
	Vector3f coeff; Vector3i pos; TO_INT_FLOOR3(pos, coeff, point);

	int vmIndex;
	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 0), direction, vmIndex, cache);
	color += (1.0f - coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 0), direction, vmIndex, cache);
	color += (coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 0), direction, vmIndex, cache);
	color += (1.0f - coeff.x) * (coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 0), direction, vmIndex, cache);
	color += (coeff.x) * (coeff.y) * (1.0f - coeff.z) * voxel.clr.toFloat();

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 1), direction, vmIndex, cache);
	color += (1.0f - coeff.x) * (1.0f - coeff.y) * coeff.z * voxel.clr.toFloat();

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 1), direction, vmIndex, cache);
	color += (coeff.x) * (1.0f - coeff.y) * coeff.z * voxel.clr.toFloat();

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 1), direction, vmIndex, cache);
	color += (1.0f - coeff.x) * (coeff.y) * coeff.z * voxel.clr.toFloat();

	voxel = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 1), direction, vmIndex, cache);
	color += (coeff.x) * (coeff.y) * coeff.z * voxel.clr.toFloat();

	return Vector4f(color / 255.0f, 255.0f);
}

//template<class TVoxel, class TIndex, class TCache>
//_CPU_AND_GPU_CODE_ inline Vector4f readFromSDF_color4u_interpolated(const CONSTPTR(TVoxel) *voxelData,
//	const CONSTPTR(TIndex) *voxelIndex, const THREADPTR(Vector3f) & point, const TSDFDirection direction,
//	THREADPTR(TCache) & cache, int & maxW)
//{
//	TVoxel resn; Vector3f ret(0.0f); Vector4f ret4; int vmIndex;
//	Vector3f coeff; Vector3i pos; TO_INT_FLOOR3(pos, coeff, point);
//
//	resn = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 0), direction, vmIndex, cache);
//	maxW = resn.w_depth;
//	ret += (1.0f - coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * resn.clr.toFloat();
//
//	resn = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 0), direction, vmIndex, cache);
//	if (resn.w_depth > maxW) maxW = resn.w_depth;
//	ret += (coeff.x) * (1.0f - coeff.y) * (1.0f - coeff.z) * resn.clr.toFloat();
//
//	resn = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 0), direction, vmIndex, cache);
//	if (resn.w_depth > maxW) maxW = resn.w_depth;
//	ret += (1.0f - coeff.x) * (coeff.y) * (1.0f - coeff.z) * resn.clr.toFloat();
//
//	resn = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 0), direction, vmIndex, cache);
//	if (resn.w_depth > maxW) maxW = resn.w_depth;
//	ret += (coeff.x) * (coeff.y) * (1.0f - coeff.z) * resn.clr.toFloat();
//
//	resn = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 0, 1), direction, vmIndex, cache);
//	if (resn.w_depth > maxW) maxW = resn.w_depth;
//	ret += (1.0f - coeff.x) * (1.0f - coeff.y) * coeff.z * resn.clr.toFloat();
//
//	resn = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 0, 1), direction, vmIndex, cache);
//	if (resn.w_depth > maxW) maxW = resn.w_depth;
//	ret += (coeff.x) * (1.0f - coeff.y) * coeff.z * resn.clr.toFloat();
//
//	resn = readVoxel(voxelData, voxelIndex, pos + Vector3i(0, 1, 1), direction, vmIndex, cache);
//	if (resn.w_depth > maxW) maxW = resn.w_depth;
//	ret += (1.0f - coeff.x) * (coeff.y) * coeff.z * resn.clr.toFloat();
//
//	resn = readVoxel(voxelData, voxelIndex, pos + Vector3i(1, 1, 1), direction, vmIndex, cache);
//	if (resn.w_depth > maxW) maxW = resn.w_depth;
//	ret += (coeff.x) * (coeff.y) * coeff.z * resn.clr.toFloat();
//
//	ret4.x = ret.x; ret4.y = ret.y; ret4.z = ret.z; ret4.w = 255.0f;
//
//	return ret4 / 255.0f;
//}

#define ReadVoxelSDFDiscardZero(dst, pos) voxel = readVoxel(voxelData, voxelIndex, pos, direction, vmIndex); if (voxel.w_depth == 0) return Vector3f(0, 0, 0); dst = voxel.sdf;
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
template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline Vector3f computeGradientFromSDF(const TVoxel* voxelData, const TIndex* voxelIndex,
                                                          const Vector3f& point, const TSDFDirection direction)
{
	int vmIndex;

	TVoxel voxel;

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

	ret.x = TVoxel::valueToFloat(p1 * ncoeff.x + p2 * coeff.x - v1);

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

	ret.y = TVoxel::valueToFloat(p1 * ncoeff.y + p2 * coeff.y - v1);

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

	ret.z = TVoxel::valueToFloat(p1 * ncoeff.z + p2 * coeff.z - v1);
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
template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline Vector3f computeSingleNormalFromSDF(const TVoxel *voxelData,
                                                              const TIndex *voxelIndex, const Vector3f &point,
                                                              const TSDFDirection direction, const float tau=0.0)
{
	Vector3f gradient = computeGradientFromSDF(voxelData, voxelIndex, point, direction);

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

template<bool hasColor, class TVoxel, class TIndex> struct VoxelColorReader;

template<class TVoxel, class TIndex>
struct VoxelColorReader<false, TVoxel, TIndex> {
	_CPU_AND_GPU_CODE_ static Vector4f interpolate(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
		const THREADPTR(Vector3f) & point, const TSDFDirection direction)
	{
		return Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
	}
};

template<class TVoxel, class TIndex>
struct VoxelColorReader<true, TVoxel, TIndex> {
	_CPU_AND_GPU_CODE_ static Vector4f interpolate(const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
		const THREADPTR(Vector3f) & point, const TSDFDirection direction)
	{
		typename TIndex::IndexCache cache;
		return readFromSDF_color4u_interpolated(voxelData, voxelIndex, point, direction, cache);
	}
};

/**
* \brief The specialisations of this struct template can be used to write/read colours to/from surfels.
*
* \tparam hasColour  Whether or not the surfel type can store colour information.
*/
template <bool hasColour> struct SurfelColourManipulator;

/**
* \brief This template specialisation can be used to write/read dummy colours to/from surfels.
*
* It is intended for use with surfel types that cannot store colour information.
*/
template <>
struct SurfelColourManipulator<false>
{
	/**
	* \brief Simulates the reading of a colour from the specified surfel.
	*
	* \param surfel  The surfel.
	* \return        A dummy colour (black).
	*/
	template <typename TSurfel>
	_CPU_AND_GPU_CODE_
		static Vector3u read(const TSurfel& surfel)
	{
		return Vector3u((uchar)0);
	}

	/**
	* \brief Simulates the writing of a colour to the specified surfel.
	*
	* In practice, this is just a no-op, since the surfel can't store a colour.
	*
	* \param surfel  The surfel.
	* \param colour  The colour.
	*/
	template <typename TSurfel>
	_CPU_AND_GPU_CODE_
		static void write(TSurfel& surfel, const Vector3u& colour)
	{
		// No-op
	}
};

/**
* \brief This template specialisation can be used to write/read actual colours to/from surfels.
*
* It is intended for use with surfel types that can store colour information.
*/
template <>
struct SurfelColourManipulator<true>
{
	/**
	* \brief Gets the colour of the specified surfel.
	*
	* \param surfel  The surfel.
	* \return        The surfel's colour.
	*/
	template <typename TSurfel>
	_CPU_AND_GPU_CODE_
		static Vector3u read(const TSurfel& surfel)
	{
		return surfel.colour;
	}

	/**
	* \brief Sets the colour of the specified surfel.
	*
	* \param surfel  The surfel.
	* \param colour  The surfel's new colour.
	*/
	template <typename TSurfel>
	_CPU_AND_GPU_CODE_
		static void write(TSurfel& surfel, const Vector3u& colour)
	{
		surfel.colour = colour;
	}
};

} // namespace ITMLIb
