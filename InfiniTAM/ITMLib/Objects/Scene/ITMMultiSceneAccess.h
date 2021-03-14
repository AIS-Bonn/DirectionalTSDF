// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMLib/Objects/Scene/ITMDirectional.h"
#include "../../Objects/Scene/ITMRepresentationAccess.h"

#define MAX_NUM_LOCALMAPS 32

namespace ITMLib {
	struct ITMMultiCache {};

	template<class TIndex>
	class ITMMultiIndex
	{
	public:
		typedef TIndex IndexType;
		typedef ITMMultiCache IndexCache;

		struct IndexData
		{
			int numLocalMaps;
			typedef TIndex IndexType;
			typename TIndex::IndexData *index[MAX_NUM_LOCALMAPS];
			Matrix4f poses_vs[MAX_NUM_LOCALMAPS];
			Matrix4f posesInv[MAX_NUM_LOCALMAPS];
		};
	};

	template<class TVoxel>
	class ITMMultiVoxel
	{
	public:
		typedef TVoxel VoxelType;
		TVoxel* voxels[MAX_NUM_LOCALMAPS];

		static const CONSTPTR(bool) hasColorInformation = TVoxel::hasColorInformation;
	};

template<class TMultiVoxel, class TMultiIndex>
_CPU_AND_GPU_CODE_ inline float readFromSDF_float_uninterpolated(const TMultiVoxel *voxelData, const TMultiIndex *voxelIndex, const Vector3f & point, const TSDFDirection direction, int & vmIndex, ITMMultiCache & _cache)
{
	typedef typename TMultiVoxel::VoxelType TVoxel;
	typedef typename TMultiIndex::IndexType TIndex;

	float sum_sdf = 0.0f;
	int sum_weights = 0;
	vmIndex = false;
	for (int localMapId = 0; localMapId < voxelIndex->numLocalMaps; ++localMapId)
	{
		Vector3f point_local = voxelIndex->poses_vs[localMapId] * point;

		int vmIndex_tmp;
		typename TIndex::IndexCache cache;
		const TVoxel & v = readVoxel(voxelData->voxels[localMapId], voxelIndex->index[localMapId], Vector3i((int)ROUND(point_local.x), (int)ROUND(point_local.y), (int)ROUND(point_local.z)), direction, vmIndex_tmp, cache);
		if (!vmIndex_tmp) continue;

		vmIndex = true;
		sum_sdf += (float)v.w_depth * (float)v.sdf;
		sum_weights += v.w_depth;
	}
	if (sum_weights == 0) return 1.0f;
	return TVoxel::valueToFloat(sum_sdf / (float)sum_weights);
}

template<class TMultiVoxel, class TMultiIndex>
_CPU_AND_GPU_CODE_ inline float readFromSDF_float_interpolated(const TMultiVoxel *voxelData, const TMultiIndex *voxelIndex, const Vector3f & point, const TSDFDirection direction, int & vmIndex, ITMMultiCache & _cache)
{
	typedef typename TMultiIndex::IndexType TIndex;

	float sum_sdf = 0.0f;
	int sum_weights = 0;
	vmIndex = false;

	for (int localMapId = 0; localMapId < voxelIndex->numLocalMaps; ++localMapId) 
	{
		Vector3f point_local = voxelIndex->poses_vs[localMapId] * point;

		int vmIndex_tmp, maxW;
		typename TIndex::IndexCache cache;
		
		float sdf = readFromSDF_float_interpolated(voxelData->voxels[localMapId], voxelIndex->index[localMapId],
			point_local, direction, vmIndex_tmp, cache, maxW);
		if (!vmIndex_tmp) continue;
		
		vmIndex = true;

		sum_sdf += (float)maxW * sdf;
		sum_weights += maxW;
	}

	if (sum_weights == 0) return 1.0f;
	
	return (sum_sdf / (float)sum_weights);
}

template<class TMultiVoxel, class TMultiIndex>
_CPU_AND_GPU_CODE_ inline Vector4f readFromSDF_color4u_uninterpolated(float &confidence,  const TMultiVoxel *voxelData, const TMultiIndex *voxelIndex, const Vector3f & point, const TSDFDirection direction, ITMMultiCache & _cache, const int maxW)
{
	typedef typename TMultiIndex::IndexType TIndex;

	confidence = 0;
	Vector4f accu(0.0f);
	for (int localMapId = 0; localMapId < voxelIndex->numLocalMaps; ++localMapId)
	{
		Vector3f point_local = voxelIndex->poses_vs[localMapId] * point;

		int maxW;
		typename TIndex::IndexCache cache;
		float conf_;
		Vector4f val = readFromSDF_color4u_uninterpolated(conf_, voxelData->voxels[localMapId], voxelIndex->index[localMapId], point_local, direction, cache, maxW);

		accu += (float)maxW * val;
		confidence += conf_;
	}
	if (accu.w < 0.001f) accu.w = 1.0f;
	return (accu / accu.w);
}

template<class TMultiVoxel, class TMultiIndex>
_CPU_AND_GPU_CODE_ inline Vector4f readFromSDF_color4u_interpolated(const TMultiVoxel *voxelData, const TMultiIndex *voxelIndex, const Vector3f & point, const TSDFDirection direction, ITMMultiCache & _cache)
{
	typedef typename TMultiIndex::IndexType TIndex;

	Vector4f accu(0.0f);
	for (int localMapId = 0; localMapId < voxelIndex->numLocalMaps; ++localMapId)
	{
		Vector3f point_local = voxelIndex->poses_vs[localMapId] * point;

		int maxW;
		typename TIndex::IndexCache cache;
		float conf_;
		Vector4f val = readFromSDF_color4u_interpolated(voxelData->voxels[localMapId], voxelIndex->index[localMapId], point_local, direction, cache);

		accu += (float)maxW * val;
	}
	if (accu.w < 0.001f) accu.w = 1.0f;
	return (accu / accu.w);
}

template<class TMultiVoxel, class TMultiIndex>
_CPU_AND_GPU_CODE_ inline Vector4f readFromSDF_color4u_interpolated(float &confidence,  const TMultiVoxel *voxelData, const TMultiIndex *voxelIndex, const Vector3f & point, const TSDFDirection direction, ITMMultiCache & _cache, const int maxW)
{
	typedef typename TMultiIndex::IndexType TIndex;

	confidence = 0;
	Vector4f accu(0.0f);
	for (int localMapId = 0; localMapId < voxelIndex->numLocalMaps; ++localMapId) 
	{
		Vector3f point_local = voxelIndex->poses_vs[localMapId] * point;

		int maxW;
		typename TIndex::IndexCache cache;
		float conf_;
		Vector4f val = readFromSDF_color4u_interpolated(conf_, voxelData->voxels[localMapId], voxelIndex->index[localMapId], point_local, direction, cache, maxW);

		accu += (float)maxW * val;
		confidence += conf_;
	}
	if (accu.w < 0.001f) accu.w = 1.0f;
	return (accu / accu.w);
}

template<class TMultiVoxel, class TMultiIndex>
_CPU_AND_GPU_CODE_ inline float readWithConfidenceFromSDF_float_uninterpolated(THREADPTR(float) &confidence, const CONSTPTR(TMultiVoxel) *voxelData,
                                                                               const CONSTPTR(TMultiIndex) *voxelIndex, Vector3f point, const TSDFDirection direction, const int maxW, THREADPTR(int) &vmIndex, ITMMultiCache & _cache)
{
	typedef typename TMultiVoxel::VoxelType TVoxel;
	typedef typename TMultiIndex::IndexType TIndex;

	float sum_sdf = 0.0f;
	int sum_weights = 0;
	vmIndex = false;
	for (int localMapId = 0; localMapId < voxelIndex->numLocalMaps; ++localMapId)
	{
		Vector3f point_local = voxelIndex->poses_vs[localMapId] * point;

		int vmIndex_tmp;
		typename TIndex::IndexCache cache;
		const TVoxel & v = readVoxel(voxelData->voxels[localMapId], voxelIndex->index[localMapId], Vector3i((int)ROUND(point_local.x), (int)ROUND(point_local.y), (int)ROUND(point_local.z)), direction, vmIndex_tmp, cache);
		if (!vmIndex_tmp) continue;

		vmIndex = true;
		sum_sdf += (float)v.w_depth * (float)v.sdf;
		sum_weights += v.w_depth;
	}
	confidence = TVoxel::weightToFloat(sum_weights, maxW);
	if (sum_weights == 0) return 1.0f;

	return TVoxel::valueToFloat(sum_sdf / (float)sum_weights);
}

template<class TMultiVoxel, class TMultiIndex>
_CPU_AND_GPU_CODE_ inline float readWithConfidenceFromSDF_float_interpolated(THREADPTR(float) &confidence, const CONSTPTR(TMultiVoxel) *voxelData,
	const CONSTPTR(TMultiIndex) *voxelIndex, Vector3f point, const TSDFDirection direction, const int maxW, THREADPTR(int) &vmIndex, ITMMultiCache & _cache)
{
	typedef typename TMultiIndex::IndexType TIndex;

	float sum_sdf = 0.0f, sum_confidence = 0.0f;
	int noLiveScenes = 0;
	
	vmIndex = false;
	for (int localMapId = 0; localMapId < voxelIndex->numLocalMaps; ++localMapId) 
	{
		Vector3f point_local = voxelIndex->poses_vs[localMapId] * point;

		int vmIndex_tmp;
		typename TIndex::IndexCache cache;

		float conf;
		float sdf = readWithConfidenceFromSDF_float_interpolated(conf, voxelData->voxels[localMapId], voxelIndex->index[localMapId], point_local, direction, maxW, vmIndex_tmp, cache);
		
		if (!vmIndex_tmp) continue;
		vmIndex = true;

		sum_sdf += (float)conf * sdf;
		sum_confidence += conf;
		noLiveScenes++;
	}

	if (sum_confidence == 0.0f) { confidence = 1.0f;  return 1.0f; }

	confidence = sum_confidence / (float)noLiveScenes + 1.0f;
	return sum_sdf / (float)sum_confidence;
}

template<class TVoxel, class TMultiIndex>
_CPU_AND_GPU_CODE_ inline Vector3f computeSingleNormalFromSDF(const ITMMultiVoxel<TVoxel> *voxelData,
                                                              const TMultiIndex *voxelIndex, const Vector3f &point,
                                                              const TSDFDirection direction, const float tau=0.0)
{
//	typedef typename TMultiIndex::IndexType TIndex;

	printf("Not implemented: %s\n", __PRETTY_FUNCTION__);

	return Vector3f(0, 0, 0);
}

}
