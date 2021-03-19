// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../../Objects/Tracking/ITMTrackingState.h"
#include "../../../Objects/Views/ITMView.h"

#include "../Shared/ITMVisualisationEngine_Shared.h"
#include "../../../Utils/ITMCUDAUtils.h"
#include "RenderingTSDF_CUDA.cuh"

#include <stdgpu/unordered_set_fwd>
#include <stdgpu/unordered_map.cuh>

namespace ITMLib
{
	// declaration of device functions

	__global__ void buildCompleteVisibleList_device(const ITMHashEntry *hashTable, /*ITMHashCacheState *cacheStates, bool useSwapping,*/ int noTotalEntries,
		int *visibleEntryIDs, int *noVisibleEntries, HashEntryVisibilityType *entriesVisibleType, Matrix4f M, Vector4f projParams, Vector2i imgSize, float voxelSize);

	__global__ void countVisibleBlocks_device(const int *visibleEntryIDs, int noVisibleEntries, const ITMHashEntry *hashTable, uint *noBlocks, int minBlockId, int maxBlockId);

	__global__ void projectAndSplitBlocks_device(const ITMHashEntry *hashEntries, const int *visibleEntryIDs, int noVisibleEntries,
		const Matrix4f pose_M, const Vector4f intrinsics, const Vector2i imgSize, float voxelSize, RenderingBlock *renderingBlocks,
		uint *noTotalBlocks);

	__global__ void checkProjectAndSplitBlocks_device(const ITMHashEntry *hashEntries, int noHashEntries,
		const Matrix4f pose_M, const Vector4f intrinsics, const Vector2i imgSize, float voxelSize, RenderingBlock *renderingBlocks,
		uint *noTotalBlocks);

	__global__ void computeMinMaxData_device(uint noTotalBlocks, const RenderingBlock *renderingBlocks,
	                                         Vector2i imgSize, Vector2f *minmaxData);

	__global__ void findMissingPoints_device(int *fwdProjMissingPoints, uint *noMissingPoints, const Vector2f *minmaximg,
		Vector4f *forwardProjection, float *currentDepth, Vector2i imgSize);

	__global__ void forwardProject_device(Vector4f *forwardProjection, const Vector4f *pointsRay, Vector2i imgSize, Matrix4f M,
		Vector4f projParams, float voxelSize);

__global__ void findVisibleBlocks_device(RenderingTSDF tsdf, const ITMHashEntry* hashTable,
                                         int noTotalEntries, Matrix4f M, Vector4f projParams, Vector2i imgSize,
                                         float voxelSize);

__device__ inline bool castRayDefaultRenderingTSDF(Vector4f& pt_out,
                                              float& distance_out,
                                              int x, int y, RenderingTSDF tsdf,
                                              Matrix4f invM, Vector4f invProjParams,
                                              const ITMSceneParams& sceneParams,
                                              const CONSTPTR(Vector2f)& minMaxImg,
                                              float maxDistance = -1)
{
	Vector4f pt_camera_f;
	Vector3f pt_block_s, pt_block_e, rayDirection, pt_result;
	float sdfValue = 1.0f, confidence = 0;
	float totalLength, stepLength, totalLengthMax, stepScale;

	pt_out = Vector4f(0, 0, 0, 0);

	stepScale = sceneParams.mu * sceneParams.oneOverVoxelSize;

	pt_camera_f = Vector4f(reprojectImagePoint(x, y, minMaxImg.x, invProjParams), 1.0f);
	totalLength = length(TO_VECTOR3(pt_camera_f)) * sceneParams.oneOverVoxelSize;
	pt_block_s = TO_VECTOR3(invM * pt_camera_f) * sceneParams.oneOverVoxelSize;

	pt_camera_f = Vector4f(reprojectImagePoint(x, y,
	                                           maxDistance > 0 ? maxDistance : minMaxImg.y,
	                                           invProjParams), 1.0f);
	totalLengthMax = length(TO_VECTOR3(pt_camera_f)) * sceneParams.oneOverVoxelSize;
	pt_block_e = TO_VECTOR3(invM * pt_camera_f) * sceneParams.oneOverVoxelSize;

	rayDirection = (pt_block_e - pt_block_s).normalised();

	pt_result = pt_block_s;

//	bool secondTry = false;
//	foo:
	bool found;
	float lastSDFValue = sdfValue;
	while (totalLength < totalLengthMax)
	{
		sdfValue = readFromSDF_float_uninterpolated(found, tsdf, pt_result);

		if (!found)
		{
			stepLength = SDF_BLOCK_SIZE;
		} else
		{
			if ((sdfValue <= 0.1f) && (sdfValue >= -0.5f))
			{
				sdfValue = readFromSDF_float_interpolated(found, tsdf, pt_result);
			}
			if (lastSDFValue > 0 and sdfValue <= 0)// and lastSDFValue - sdfValue < 1.5)
			{
				break;
			}
			stepLength = MAX(sdfValue * stepScale, 1.0f);
		}

		lastSDFValue = sdfValue;
		pt_result += stepLength * rayDirection;
		totalLength += stepLength;
	}

	if (sdfValue > 0.0f)
		return false;

// Perform 2 additional steps to get more accurate zero crossing
	for (int i = 0; i < 2 or (i < 5 and fabs(sdfValue) > 1e-6); i++)
	{
		lastSDFValue = sdfValue;
		stepLength = sdfValue * stepScale;
		pt_result += stepLength * rayDirection;
		totalLength += stepLength;

		sdfValue = readWithConfidenceFromSDF_float_interpolated(found, confidence, tsdf, pt_result, sceneParams.maxW);

// Compensate sign hopping with little to no reduction in magnitude (steep angles)
		float reductionFactor = (fabs(sdfValue) - fabs(lastSDFValue)) / fabs(lastSDFValue);
// rF < 0: magnitude reduction, rF > 0: magnitude increase
		if (SIGN(sdfValue) != SIGN(lastSDFValue) and reductionFactor > -0.75)
		{
			stepScale *= 0.5;
		}
	}

	if (fabs(sdfValue) > 0.1)
	{
//		totalLength += 2;
//		pt_result += 2 * rayDirection;
//		if (not secondTry)
//		{
//			secondTry = true;
//			goto foo;
//		}
		return false;
	}

	distance_out = totalLength / sceneParams.oneOverVoxelSize;
// multiply by transition: negative transition <=> negative confidence!
	pt_out = Vector4f(pt_result, confidence);

	return true;
}


template<class TVoxel, class TIndex>
__device__ inline bool castRayDefaultCombinedTSDF(DEVICEPTR(Vector4f)& pt_out,
                                              float& distance_out,
                                              int x, int y, const CONSTPTR(TVoxel)* voxelData,
                                              const CONSTPTR(typename TIndex::IndexData)* voxelIndex,
                                              stdgpu::unordered_map<Vector3s, Vector6f*> combinedTSDF,
                                              Matrix4f invM, Vector4f invProjParams,
                                              const ITMSceneParams& sceneParams,
                                              const CONSTPTR(Vector2f)& minMaxImg,
                                              const TSDFDirection direction = TSDFDirection::NONE,
                                              float maxDistance = -1)
{
	Vector4f pt_camera_f;
	Vector3f pt_block_s, pt_block_e, rayDirection, pt_result;
	int vmIndex = 0;
	float sdfValue = 1.0f, confidence = 0;
	float totalLength, stepLength, totalLengthMax, stepScale;

	pt_out = Vector4f(0, 0, 0, 0);

	stepScale = sceneParams.mu * sceneParams.oneOverVoxelSize;

	pt_camera_f = Vector4f(reprojectImagePoint(x, y, minMaxImg.x, invProjParams), 1.0f);
	totalLength = length(TO_VECTOR3(pt_camera_f)) * sceneParams.oneOverVoxelSize;
	pt_block_s = TO_VECTOR3(invM * pt_camera_f) * sceneParams.oneOverVoxelSize;

	pt_camera_f = Vector4f(reprojectImagePoint(x, y,
	                                           maxDistance > 0 ? maxDistance : minMaxImg.y,
	                                           invProjParams), 1.0f);
	totalLengthMax = length(TO_VECTOR3(pt_camera_f)) * sceneParams.oneOverVoxelSize;
	pt_block_e = TO_VECTOR3(invM * pt_camera_f) * sceneParams.oneOverVoxelSize;

	rayDirection = (pt_block_e - pt_block_s).normalised();

	pt_result = pt_block_s;

	typename TIndex::IndexCache cache;


//	bool secondTry = false;
//	foo:
	float lastSDFValue = sdfValue;
	while (totalLength < totalLengthMax)
	{
		Vector3i voxelIdx((int)ROUND(pt_result.x), (int)ROUND(pt_result.y), (int)ROUND(pt_result.z));
		Vector3i blockIdx;
		unsigned short offset;
		voxelToBlockPosAndOffset(voxelIdx, blockIdx, offset);
		auto it = combinedTSDF.find(blockIdx.toShort());

		vmIndex = false;
		if (it != combinedTSDF.end())
		{
			const Vector6f& weigths = it->second[offset];
			float weightSum = 0;
			float sdfSum = 0;
			for (int directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
			{
				if (weigths.v[directionIdx] <= 0)
					continue;

				int found;
				float sdf = readFromSDF_float_uninterpolated(voxelData, voxelIndex, pt_result, TSDFDirection(directionIdx), found, cache);

				if (found)
				{
					weightSum += weigths.v[directionIdx];
					sdfSum += weigths.v[directionIdx] * sdf;
				}
			}
			if (weightSum > 0)
			{
				sdfValue = sdfSum / weightSum;
				vmIndex = true;
			}
		}
//		sdfValue = readFromSDF_float_uninterpolated(voxelData, voxelIndex, pt_result, direction, vmIndex, cache);

		if (!vmIndex)
		{
			stepLength = SDF_BLOCK_SIZE;
		} else
		{
			if ((sdfValue <= 0.1f) && (sdfValue >= -0.5f))
			{
				vmIndex = false;
				if (it != combinedTSDF.end())
				{
					const Vector6f& weigths = it->second[offset];
					float weightSum = 0;
					float sdfSum = 0;
					for (int directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
					{
						if (weigths.v[directionIdx] <= 0)
							continue;

						int found;
						float sdf = readFromSDF_float_interpolated(voxelData, voxelIndex, pt_result, TSDFDirection(directionIdx), found, cache);

						if (found)
						{
							weightSum += weigths.v[directionIdx];
							sdfSum += weigths.v[directionIdx] * sdf;
						}
					}
					if (weightSum > 0)
					{
						sdfValue = sdfSum / weightSum;
						vmIndex = true;
					}
				}
//				sdfValue = readFromSDF_float_interpolated(voxelData, voxelIndex, pt_result, direction, vmIndex, cache);
			}
			if (lastSDFValue > 0 and sdfValue <= 0)// and lastSDFValue - sdfValue < 1.5)
			{
				break;
			}
			stepLength = MAX(sdfValue * stepScale, 1.0f);
		}

		lastSDFValue = sdfValue;
		pt_result += stepLength * rayDirection;
		totalLength += stepLength;
	}

	if (sdfValue > 0.0f)
		return false;

// Perform 2 additional steps to get more accurate zero crossing
	for (int i = 0; i < 2 or (i < 5 and fabs(sdfValue) > 1e-6); i++)
	{
		lastSDFValue = sdfValue;
		stepLength = sdfValue * stepScale;
		pt_result += stepLength * rayDirection;
		totalLength += stepLength;

//		sdfValue = readWithConfidenceFromSDF_float_interpolated(confidence, voxelData, voxelIndex,
//		                                                        pt_result, direction, sceneParams.maxW,
//		                                                        vmIndex, cache);
		Vector3i voxelIdx((int)ROUND(pt_result.x), (int)ROUND(pt_result.y), (int)ROUND(pt_result.z));
		Vector3i blockIdx;
		unsigned short offset;
		voxelToBlockPosAndOffset(voxelIdx, blockIdx, offset);
		auto it = combinedTSDF.find(blockIdx.toShort());

		vmIndex = false;
		if (it != combinedTSDF.end())
		{
			const Vector6f& weigths = it->second[offset];
			float weightSum = 0;
			float sdfSum = 0;
			float confidenceSum = 0;
			for (int directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
			{
				if (weigths.v[directionIdx] <= 0)
					continue;

				int found;
				float conf;
				float sdf = readWithConfidenceFromSDF_float_interpolated(conf, voxelData, voxelIndex,
		                                                        pt_result, TSDFDirection(directionIdx), sceneParams.maxW,
		                                                        found, cache);

				if (found)
				{
					weightSum += weigths.v[directionIdx];
					sdfSum += weigths.v[directionIdx] * sdf;
					confidenceSum += weigths.v[directionIdx] * conf;
				}
			}
			if (weightSum > 0)
			{
				sdfValue = sdfSum / weightSum;
				confidence = confidenceSum / weightSum;
				vmIndex = true;
			}
		}

// Compensate sign hopping with little to no reduction in magnitude (steep angles)
		float reductionFactor = (fabs(sdfValue) - fabs(lastSDFValue)) / fabs(lastSDFValue);
// rF < 0: magnitude reduction, rF > 0: magnitude increase
		if (SIGN(sdfValue) != SIGN(lastSDFValue) and reductionFactor > -0.75)
		{
			stepScale *= 0.5;
		}
	}

	if (fabs(sdfValue) > 0.1)
	{
//		totalLength += 2;
//		pt_result += 2 * rayDirection;
//		if (not secondTry)
//		{
//			secondTry = true;
//			goto foo;
//		}
		return false;
	}

	distance_out = totalLength / sceneParams.oneOverVoxelSize;
// multiply by transition: negative transition <=> negative confidence!
	pt_out = Vector4f(pt_result, confidence);

	return true;
}


template<class TVoxel, class TIndex>
__global__ void genericRaycastCombinedTSDF_device(Vector4f *out_ptsRay, Vector6f *raycastDirectionalContribution, HashEntryVisibilityType *entriesVisibleType, const TVoxel *voxelData,
                                                  const typename TIndex::IndexData *voxelIndex,
                                                  stdgpu::unordered_map<Vector3s, Vector6f*> combinedTSDF,
                                                  Vector2i imgSize, Matrix4f invM, Vector4f invProjParams,
                                                  const ITMSceneParams sceneParams, const Vector2f *minmaximg, bool directionalTSDF)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	int locId = x + y * imgSize.x;
	int locId2 = (int)floor((float)x / minmaximg_subsample) + (int)floor((float)y / minmaximg_subsample) * imgSize.x;

	float distance;
	castRayDefaultCombinedTSDF<TVoxel, TIndex>(out_ptsRay[locId], distance, x, y, voxelData, voxelIndex,
																						combinedTSDF, invM, invProjParams, sceneParams, minmaximg[locId2]);
}

template<class TVoxel, class TIndex>
__global__ void genericRaycastRenderingTSDF_device(Vector4f *out_ptsRay, const RenderingTSDF tsdf,
																									 Vector2i imgSize, Matrix4f invM, Vector4f invProjParams,
																									 const ITMSceneParams sceneParams, const Vector2f* minmaximg)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	int locId = x + y * imgSize.x;
	int locId2 = (int)floor((float)x / minmaximg_subsample) + (int)floor((float)y / minmaximg_subsample) * imgSize.x;

	float distance;
	castRayDefaultRenderingTSDF(out_ptsRay[locId], distance, x, y, tsdf, invM, invProjParams, sceneParams, minmaximg[locId2]);
}

template<class TVoxel, class TIndex>
	__global__ void genericRaycast_device(Vector4f *out_ptsRay, Vector6f *raycastDirectionalContribution, HashEntryVisibilityType *entriesVisibleType, const TVoxel *voxelData,
		const typename TIndex::IndexData *voxelIndex, Vector2i imgSize, Matrix4f invM, Vector4f invProjParams,
		const ITMSceneParams sceneParams, const Vector2f *minmaximg, bool directionalTSDF)
	{
		int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

		if (x >= imgSize.x || y >= imgSize.y) return;

		int locId = x + y * imgSize.x;
		int locId2 = (int)floor((float)x / minmaximg_subsample) + (int)floor((float)y / minmaximg_subsample) * imgSize.x;

		castRay<TVoxel, TIndex>(out_ptsRay[locId], &raycastDirectionalContribution[locId], entriesVisibleType, x, y, voxelData, voxelIndex, invM, invProjParams,
			sceneParams, minmaximg[locId2], directionalTSDF);
	}

template<class TVoxel, class TIndex>
__global__ void genericRaycast_device(Vector4f* out_ptsRay, Vector6f* raycastDirectionalContribution,
                                      HashEntryVisibilityType* entriesVisibleType, const TVoxel* voxelData,
                                      const typename TIndex::IndexData* voxelIndex, Vector2i imgSize,
                                      Matrix4f invM, Vector4f invProjParams,
                                      const ITMSceneParams sceneParams, const Vector2f* minmaximg,
                                      TSDFDirection direction)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	int locId = x + y * imgSize.x;
	int locId2 = (int)floor((float)x / minmaximg_subsample) + (int)floor((float)y / minmaximg_subsample) * imgSize.x;

	float distance;
	castRayDefault<TVoxel, TIndex>(out_ptsRay[locId], distance, entriesVisibleType, x, y, voxelData, voxelIndex, invM, invProjParams,
																					sceneParams, minmaximg[locId2], direction);
}


template<class TVoxel, class TIndex>
__global__ void combineDirectionalPointClouds_device(Vector4f* out_ptsRay, Vector4f* out_normals, const InputPointClouds in_ptsRay,
                                                     Vector6f* raycastDirectionalContribution,
                                                     HashEntryVisibilityType* entriesVisibleType,
                                                     const TVoxel* voxelData,
                                                     const typename TIndex::IndexData* voxelIndex, Vector2i imgSize,
                                                     Matrix4f invM, Vector4f invProjParams,
                                                     const ITMSceneParams sceneParams, const Vector2f* minmaximg)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x);
	int y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	combineDirectionalPointClouds<true, false>(out_ptsRay, out_normals, in_ptsRay, raycastDirectionalContribution, imgSize,
		invM, invProjParams, x, y, sceneParams.voxelSize);
}

	template<class TVoxel, class TIndex>
	__global__ void genericRaycastMissingPoints_device(Vector4f *forwardProjection, Vector6f *raycastDirectionalContribution, HashEntryVisibilityType *entriesVisibleType, const TVoxel *voxelData,
		const typename TIndex::IndexData *voxelIndex, Vector2i imgSize, Matrix4f invM, Vector4f invProjParams,
		int *fwdProjMissingPoints, int noMissingPoints, const ITMSceneParams sceneParams, const Vector2f *minmaximg, bool directionalTSDF)
	{
		int pointId = threadIdx.x + blockIdx.x * blockDim.x;

		if (pointId >= noMissingPoints) return;

		int locId = fwdProjMissingPoints[pointId];
		int y = locId / imgSize.x, x = locId - y*imgSize.x;
		int locId2 = (int)floor((float)x / minmaximg_subsample) + (int)floor((float)y / minmaximg_subsample) * imgSize.x;

		castRay<TVoxel, TIndex>(forwardProjection[locId], &raycastDirectionalContribution[locId], entriesVisibleType, x, y, voxelData, voxelIndex, invM, invProjParams, sceneParams, minmaximg[locId2], directionalTSDF);
	}

	template<bool flipNormals>
	__global__ void renderICP_device(Vector4f *pointsMap, Vector4f *normalsMap, const Vector4f *pointsRay, const Vector4f *normalsRay,
		float voxelSize, Vector2i imgSize, Vector3f lightSource)
	{
		int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

		if (x >= imgSize.x || y >= imgSize.y) return;

		processPixelICP<true, flipNormals>(pointsMap, normalsMap, pointsRay, normalsRay, imgSize, x, y, voxelSize, lightSource);
	}

	template<bool flipNormals>
	__global__ void renderGrey_ImageNormals_device(Vector4u *outRendering, const Vector4f *pointsRay, const Vector4f *normalsRay, float voxelSize, Vector2i imgSize, Vector3f lightSource)
	{
		int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

		if (x >= imgSize.x || y >= imgSize.y) return;

		processPixelGrey_ImageNormals<true, flipNormals>(outRendering, pointsRay, normalsRay, imgSize, x, y, voxelSize, lightSource);
	}

	template<bool flipNormals>
	__global__ void renderNormals_ImageNormals_device(Vector4u *outRendering, const Vector4f *ptsRay, const Vector4f *normalsRay, Vector2i imgSize, float voxelSize, Vector3f lightSource)
	{
		int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

		if (x >= imgSize.x || y >= imgSize.y) return;

		processPixelNormals_ImageNormals<true, flipNormals>(outRendering, ptsRay, normalsRay, imgSize, x, y, voxelSize, lightSource);
	}

	template<bool flipNormals>
	__global__ void renderConfidence_ImageNormals_device(Vector4u *outRendering, const Vector4f *ptsRay,
		const Vector4f *normalsRay, Vector2i imgSize,
		const ITMSceneParams sceneParams, Vector3f lightSource)
	{
		int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

		if (x >= imgSize.x || y >= imgSize.y) return;

		processPixelConfidence_ImageNormals<true, flipNormals>(outRendering, ptsRay, normalsRay, imgSize, x, y, sceneParams, lightSource);
	}

template<class TVoxel, class TIndex>
__device__ inline void computeNormalAndAngle(THREADPTR(bool)& foundPoint, const THREADPTR(Vector3f)& point,
                                                     const RenderingTSDF tsdf,
                                                     const THREADPTR(Vector3f)& lightSource,
                                                     THREADPTR(Vector3f)& outNormal, THREADPTR(float)& angle)
{
	if (!foundPoint) return;

	outNormal = computeSingleNormalFromSDF(tsdf, point).normalised();

	Vector3f lightDirection = (lightSource - point).normalised();
	angle = dot(outNormal, lightDirection);
	if (!(angle > 0.0)) foundPoint = false;
}

template<class TVoxel, class TIndex>
__device__ inline void processPixelGrey_SDFNormals(DEVICEPTR(Vector4u)& outRendering, const CONSTPTR(Vector3f)& point,
                                                           bool foundPoint, const RenderingTSDF tsdf,
                                                           Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	computeNormalAndAngle<TVoxel, TIndex>(foundPoint, point, tsdf, lightSource, outNormal, angle);

	if (foundPoint) drawPixelGrey(outRendering, angle);
	else outRendering = Vector4u((uchar) 0);
}

template<class TVoxel, class TIndex>
__global__ void renderGrey_device(Vector4u *outRendering, const Vector4f *ptsRay,
                                  const RenderingTSDF tsdf, Vector2i imgSize, Vector3f lightSource)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	int locId = x + y * imgSize.x;

	Vector4f ptRay = ptsRay[locId];

	processPixelGrey_SDFNormals<TVoxel, TIndex>(outRendering[locId], ptRay.toVector3(),
																							ptRay.w > 0, tsdf, lightSource);
}


template<class TVoxel, class TIndex>
	__global__ void renderGrey_device(Vector4u *outRendering, const Vector4f *ptsRay,
		const Vector6f *directionalContribution, const TVoxel *voxelData,
		const typename TIndex::IndexData *voxelIndex, Vector2i imgSize, Vector3f lightSource)
	{
		int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

		if (x >= imgSize.x || y >= imgSize.y) return;

		int locId = x + y * imgSize.x;

		Vector4f ptRay = ptsRay[locId];

		if (directionalContribution)
			processPixelGrey_SDFNormals<TVoxel, TIndex>(outRendering[locId], ptRay.toVector3(),
			                                            &directionalContribution[locId], ptRay.w > 0, voxelData, voxelIndex,
			                                            lightSource);
		else
			processPixelGrey_SDFNormals<TVoxel, TIndex>(outRendering[locId], ptRay.toVector3(),
			                                            nullptr, ptRay.w > 0, voxelData, voxelIndex, lightSource);
	}

template<class TVoxel, class TIndex>
	__global__ void renderColourFromNormals_device(Vector4u *outRendering, const Vector4f *ptsRay,
	                                               const Vector6f *directionalContribution, const TVoxel *voxelData,
	                                               const typename TIndex::IndexData *voxelIndex, Vector2i imgSize, Vector3f lightSource)
	{
		int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

		if (x >= imgSize.x || y >= imgSize.y) return;

		int locId = x + y * imgSize.x;

		Vector4f ptRay = ptsRay[locId];

		if (directionalContribution)
			processPixelNormal_SDFNormals<TVoxel, TIndex>(outRendering[locId], ptRay.toVector3(),
			                                              &directionalContribution[locId], ptRay.w > 0, voxelData, voxelIndex,
			                                              lightSource);
		else
			processPixelNormal_SDFNormals<TVoxel, TIndex>(outRendering[locId], ptRay.toVector3(),
			                                              nullptr, ptRay.w > 0, voxelData, voxelIndex, lightSource);
	}

	template<class TVoxel, class TIndex>
	__global__ void renderColourFromConfidence_device(Vector4u *outRendering, const Vector4f *ptsRay,
		const Vector6f *directionalContribution, const TVoxel *voxelData,
		const typename TIndex::IndexData *voxelIndex, Vector2i imgSize,
		const ITMSceneParams sceneParams, Vector3f lightSource)
	{
		int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

		if (x >= imgSize.x || y >= imgSize.y) return;

		int locId = x + y * imgSize.x;

		Vector4f ptRay = ptsRay[locId];

		if (directionalContribution)
			processPixelConfidence_SDFNormals<TVoxel, TIndex>(outRendering[locId], ptRay, &directionalContribution[locId],
			                                                  ptRay.w > 0, voxelData, voxelIndex, sceneParams, lightSource);
		else
			processPixelConfidence_SDFNormals<TVoxel, TIndex>(outRendering[locId], ptRay, nullptr,
			                                                  ptRay.w > 0, voxelData, voxelIndex, sceneParams, lightSource);
	}

	template<class TVoxel, class TIndex>
	__global__ void renderPointCloud_device(/*Vector4u *outRendering, */Vector4f *locations, Vector4f *colours, uint *noTotalPoints,
		const Vector4f *ptsRay, const Vector6f *directionalContribution, const TVoxel *voxelData,
		const typename TIndex::IndexData *voxelIndex, bool skipPoints,
		float voxelSize, Vector2i imgSize, Vector3f lightSource)
	{
		__shared__ bool shouldPrefix;
		shouldPrefix = false;
		__syncthreads();

		bool foundPoint = false; Vector3f point(0.0f);

		int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

		if (x >= imgSize.x && y >= imgSize.y)
			return;

		int locId = x + y * imgSize.x;
		Vector3f outNormal; float angle; Vector4f pointRay;

		pointRay = ptsRay[locId];
		point = pointRay.toVector3();
		foundPoint = pointRay.w > 0;
		const Vector6f *directional = directionalContribution ? &directionalContribution[locId] : nullptr;

		computeNormalAndAngle<TVoxel, TIndex>(foundPoint, point, directional, voxelData, voxelIndex, lightSource, outNormal, angle);

		if (skipPoints && ((x % 2 == 0) || (y % 2 == 0))) foundPoint = false;

		if (foundPoint) shouldPrefix = true;

		__syncthreads();

		if (shouldPrefix)
		{
			int offset = computePrefixSum_device<uint>(foundPoint, noTotalPoints, blockDim.x * blockDim.y, threadIdx.x + threadIdx.y * blockDim.x);

			if (offset != -1)
			{
				Vector4f tmp(0, 0, 0, 0);
				if (directionalContribution)
				{
					for (TSDFDirection_type direction = 0; direction < N_DIRECTIONS; direction++)
					{
						tmp += directional->v[direction] * VoxelColorReader<TVoxel::hasColorInformation, TVoxel, TIndex>::interpolate(voxelData, voxelIndex, point, TSDFDirection(direction));
					}
				} else
				{
					tmp = VoxelColorReader<TVoxel::hasColorInformation, TVoxel, TIndex>::interpolate(voxelData, voxelIndex, point, TSDFDirection::NONE);
				}
				if (tmp.w > 0.0f) { tmp.x /= tmp.w; tmp.y /= tmp.w; tmp.z /= tmp.w; tmp.w = 1.0f; }
				colours[offset] = tmp;

				Vector4f pt_ray_out;
				pt_ray_out.x = point.x * voxelSize; pt_ray_out.y = point.y * voxelSize;
				pt_ray_out.z = point.z * voxelSize; pt_ray_out.w = 1.0f;
				locations[offset] = pt_ray_out;
			}
		}
	}

template<class TVoxel, class TIndex>
__device__ inline void processPixelColour(
	DEVICEPTR(Vector4u)& outRendering, const CONSTPTR(Vector3f)& point, bool foundPoint, const RenderingTSDF tsdf, const Vector3f lightSource)
{
	float angle;
	Vector3f outNormal;
//	computeNormalAndAngle<TVoxel, TIndex>(foundPoint, point, directionalContribution, voxelData, voxelIndex,
//	                                      lightSource, outNormal, angle);
	angle =1;

	if (foundPoint)
	{
		Vector4f clr = readFromSDF_color4u_interpolated(tsdf, point);
		outRendering.x = (uchar) (clr.x * 255.0f);
		outRendering.y = (uchar) (clr.y * 255.0f);
		outRendering.z = (uchar) (clr.z * 255.0f);
		outRendering.w = 255;

	} else outRendering = Vector4u((uchar) 0);
}

template<class TVoxel, class TIndex>
__global__ void renderColour_device(Vector4u *outRendering, const Vector4f *ptsRay,
                                    const RenderingTSDF tsdf, Vector2i imgSize, const Vector3f lightSource)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	int locId = x + y * imgSize.x;

	Vector4f ptRay = ptsRay[locId];

	processPixelColour<TVoxel, TIndex>(outRendering[locId], ptRay.toVector3(), ptRay.w > 0,
																		 tsdf, lightSource);
}

	template<class TVoxel, class TIndex>
	__global__ void renderColour_device(Vector4u *outRendering, const Vector4f *ptsRay, const Vector6f *directionalContribution,
		const TVoxel *voxelData, const typename TIndex::IndexData *voxelIndex, Vector2i imgSize, const Vector3f lightSource)
	{
		int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

		if (x >= imgSize.x || y >= imgSize.y) return;

		int locId = x + y * imgSize.x;

		Vector4f ptRay = ptsRay[locId];

		if (directionalContribution)
			processPixelColour<TVoxel, TIndex>(outRendering[locId], ptRay.toVector3(),
			                                   &directionalContribution[locId], ptRay.w > 0, voxelData, voxelIndex, lightSource);
		else
			processPixelColour<TVoxel, TIndex>(outRendering[locId], ptRay.toVector3(), nullptr, ptRay.w > 0,
			                                   voxelData, voxelIndex, lightSource);
	}

template<class TVoxel, class TIndex>
__global__ void renderDepth_device(Vector4u *outRendering, const Vector4f *ptsRay, const Matrix4f T_CW,
	const Vector2i imgSize, const float voxelSize, const float maxDepth)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	int locId = x + y * imgSize.x;

	Vector4f ptRay = ptsRay[locId];
	processPixelDepth<TVoxel, TIndex>(outRendering[locId], ptRay.toVector3(), ptRay.w > 0, T_CW, voxelSize, maxDepth);
}

template<class TVoxel, class TIndex>
__global__ void computePointCloudNormals_device(Vector4f* outputNormals, const Vector4f* pointsRay,
                                                const Vector2i imgSize, float voxelSize)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x);
	int y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	bool foundPoint = true;
	Vector3f normal;
	computeNormal<false, false>(pointsRay, voxelSize, imgSize, x, y, foundPoint, normal);

	if (not foundPoint)
	{
		outputNormals[x + y * imgSize.x] = Vector4f(0, 0, 0, -1);
		return;
	}

	outputNormals[x + y * imgSize.x] = Vector4f(normal, 1);
}


template<class TVoxel, class TIndex>
__global__ void renderPixelError_device(
	Vector4u* outRendering, const Vector4f* pointsRay, const Vector4f* normalsRay, const float* depth,
	const Matrix4f depthImagePose, const Matrix4f sceneRenderingPose, const Vector4f intrinsics,
	const Vector2i imgSize, const float maxError)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x);
	int y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	processPixelError(outRendering, pointsRay, normalsRay, depth, depthImagePose, sceneRenderingPose, intrinsics,
	                  imgSize, maxError, x, y);
}

} // ITMLib
