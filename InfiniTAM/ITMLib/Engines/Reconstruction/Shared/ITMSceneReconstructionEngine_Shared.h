// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "Objects/Scene/ITMRepresentationAccess.h"
#include "Objects/Scene/ITMDirectional.h"
#include "Utils/ITMPixelUtils.h"
#include "ITMBlockTraversal.h"
#include "ITMLib/Engines/Reconstruction/Interface/ITMSceneReconstructionEngine.h"
#include "ITMLib/Engines/Reconstruction/Shared/ITMFusionWeight.hpp"
#include "ITMLib/Utils/ITMProjectionUtils.h"

namespace ITMLib
{

struct AllocationTempData
{
	int noAllocatedVoxelEntries;
	int noAllocatedExcessEntries;
	int noVisibleEntries;
};

template<class TVoxel>
_CPU_AND_GPU_CODE_ inline float
computeUpdatedVoxelDepthInfo(DEVICEPTR(TVoxel)& voxel, const TSDFDirection direction,
                             const THREADPTR(Vector4f)& voxel_world,
                             const CONSTPTR(Matrix4f)& M_d,
                             const CONSTPTR(Vector4f)& projParams_d,
                             const CONSTPTR(ITMFusionParams)& fusionParams,
                             const CONSTPTR(ITMSceneParams)& sceneParams,
                             const CONSTPTR(float)* depth,
                             const CONSTPTR(Vector4f)* depthNormals,
                             const CONSTPTR(Vector2i)& imgSize
)
{
	Vector4f voxel_camera;
	Vector2f voxel_image;
	float depth_measure, eta, oldF, newF;
	float oldW, newW;

	// project point into image
	voxel_camera = M_d * voxel_world;
	if (voxel_camera.z <= 0) return -1;

	voxel_image = project(voxel_camera.toVector3(), projParams_d);
	if ((voxel_image.x < 1) || (voxel_image.x > imgSize.x - 2) || (voxel_image.y < 1) ||
	    (voxel_image.y > imgSize.y - 2))
		return -1;

	// get measured depth from image
	int idx = (int) (voxel_image.x + 0.5f) + (int) (voxel_image.y + 0.5f) * imgSize.x;
	depth_measure = depth[idx];
	if (depth_measure <= 0.0f) return -1;

	// check whether voxel needs updating
	eta = depth_measure - voxel_camera.z;
	if (eta < -sceneParams.mu) return eta;

	// compute updated SDF value and reliability
	oldF = TVoxel::valueToFloat(voxel.sdf);
	oldW = TVoxel::weightToFloat(voxel.w_depth, sceneParams.maxW);

	float distance = eta;
	if (fusionParams.fusionMetric == FusionMetric::FUSIONMETRIC_POINT_TO_PLANE)
	{
		Matrix4f invM_d;
		M_d.inv(invM_d);
		Vector3f pt_camera = reprojectImagePoint(voxel_image.x, voxel_image.y, depth_measure,
		                                         invertProjectionParams(projParams_d));
		Vector3f pt_world = (invM_d * Vector4f(pt_camera, 1)).toVector3();
		Vector4f normal_camera = depthNormals[idx];
		if (normal_camera.w != 1)
			return -1;
		normal_camera.w = 0; // rotation-only transformation
		Vector3f normal_world = (invM_d * normal_camera).toVector3();
		distance = ORUtils::dot(voxel_world.toVector3() - pt_world, normal_world);
	}

	newF = MIN(1.0f, distance / sceneParams.mu);
	newW = 1;
	if (fusionParams.useWeighting)
	{
		Vector4f normalCamera = depthNormals[idx];
		if (normalCamera.w != 1)
			return -1;
		Vector3f viewRay_camera = reprojectImagePoint(voxel_image.x, voxel_image.y, 1,
		                                              invertProjectionParams(projParams_d)).normalised();
		float directionWeight = 1;
		if (direction != TSDFDirection::NONE)
		{
			Matrix4f invM_d;
			M_d.inv(invM_d);
			Vector4f normalWorld = invM_d * normalCamera;
			directionWeight = DirectionWeight(TO_VECTOR3(normalWorld), direction);
			if (directionWeight < direction_weight_threshold)
				return -1;
		}
		newW = depthWeight(depth_measure, normalCamera.toVector3(), viewRay_camera, directionWeight, sceneParams);
	}

	newF = oldW * oldF + newW * newF;
	newW = oldW + newW;
	newF /= newW;
	newW = MIN(newW, sceneParams.maxW);

	// write back
	voxel.sdf = TVoxel::floatToValue(newF);
	voxel.w_depth = TVoxel::floatToWeight(newW, sceneParams.maxW);

	return distance;
}

template<class TVoxel>
_CPU_AND_GPU_CODE_ inline float
computeUpdatedVoxelDepthInfo(DEVICEPTR(TVoxel)& voxel, const TSDFDirection direction,
                             const THREADPTR(Vector4f)& voxel_world,
                             const CONSTPTR(Matrix4f)& M_d,
                             const CONSTPTR(Vector4f)& projParams_d,
                             const CONSTPTR(ITMFusionParams)& fusionParams,
                             const CONSTPTR(ITMSceneParams)& sceneParams,
                             const CONSTPTR(float)* depth,
                             const CONSTPTR(Vector4f)* depthNormals,
                             const CONSTPTR(float)* confidence, const CONSTPTR(Vector2i)& imgSize)
{
	Vector4f voxel_camera;
	Vector2f voxel_image;
	float depth_measure, eta, oldF, newF;
	int idx;
	float oldW, newW;

	// project point into image
	voxel_camera = M_d * voxel_world;
	if (voxel_camera.z <= 0) return -1;

	voxel_image = project(voxel_camera.toVector3(), projParams_d);
	if ((voxel_image.x < 1) || (voxel_image.x > imgSize.x - 2) || (voxel_image.y < 1) ||
	    (voxel_image.y > imgSize.y - 2))
		return -1;

	idx = (int) (voxel_image.x + 0.5f) + (int) (voxel_image.y + 0.5f) * imgSize.x;
	// get measured depth from image
	depth_measure = depth[idx];
	if (depth_measure <= 0.0) return -1;

	// check whether voxel needs updating
	eta = depth_measure - voxel_camera.z;
	if (eta < -sceneParams.mu) return eta;

	// compute updated SDF value and reliability
	oldF = TVoxel::valueToFloat(voxel.sdf);
	oldW = TVoxel::weightToFloat(voxel.w_depth, sceneParams.maxW);

	float distance = eta;
	if (fusionParams.fusionMetric == FusionMetric::FUSIONMETRIC_POINT_TO_PLANE)
	{
		Matrix4f invM_d;
		M_d.inv(invM_d);
		Vector3f pt_camera = reprojectImagePoint(voxel_image.x, voxel_image.y, depth_measure,
		                                         invertProjectionParams(projParams_d));
		Vector3f pt_world = (invM_d * Vector4f(pt_camera, 1)).toVector3();
		Vector4f normal_camera = depthNormals[idx];
		if (normal_camera.w != 1)
			return -1;
		normal_camera.w = 0; // rotation-only transformation
		Vector3f normal_world = (invM_d * normal_camera).toVector3();
		distance = ORUtils::dot(voxel_world.toVector3() - pt_world, normal_world);
	}

	newF = MIN(1.0f, distance / sceneParams.mu);
	newW = 1;
	if (fusionParams.useWeighting)
	{
		Vector4f normalCamera = depthNormals[idx];
		if (normalCamera.w != 1)
			return -1;
		Vector3f viewRay_camera = reprojectImagePoint(voxel_image.x, voxel_image.y, 1,
		                                              invertProjectionParams(projParams_d)).normalised();
		float directionWeight = 1;
		if (direction != TSDFDirection::NONE)
		{
			Matrix4f invM_d;
			M_d.inv(invM_d);
			Vector4f normalWorld = invM_d * normalCamera;
			directionWeight = DirectionWeight(TO_VECTOR3(normalWorld), direction);
			if (directionWeight < direction_weight_threshold)
				return -1;
		}
		newW = depthWeight(depth_measure, normalCamera.toVector3(), viewRay_camera, directionWeight, sceneParams);
	}

	newF = oldW * oldF + newW * newF;
	newW = oldW + newW;
	newF /= newW;
	newW = MIN(newW, sceneParams.maxW);

	// write back^
	voxel.sdf = TVoxel::floatToValue(newF);
	voxel.w_depth = TVoxel::floatToWeight(newW, sceneParams.maxW);
	voxel.confidence += TVoxel::floatToValue(confidence[idx]);

	return distance;
}

template<class TVoxel>
_CPU_AND_GPU_CODE_ inline void
computeUpdatedVoxelColorInfo(DEVICEPTR(TVoxel)& voxel, const TSDFDirection direction,
                             const THREADPTR(Vector4f)& pt_world,
                             const CONSTPTR(Matrix4f)& M_rgb,
                             const CONSTPTR(Vector4f)& projParams_rgb,
                             const CONSTPTR(ITMSceneParams)& sceneParams, float eta,
                             const CONSTPTR(Vector4u)* rgb, const CONSTPTR(Vector2i)& imgSize)
{
	Vector4f pt_camera;
	Vector2f pt_image;
	Vector3f rgb_measure, oldC, newC;
	Vector3u buffV3u;
	float newW, oldW;

	buffV3u = voxel.clr;
	oldW = TVoxel::weightToFloat(voxel.w_color, sceneParams.maxW);

	oldC = TO_FLOAT3(buffV3u) / 255.0f;
	newC = oldC;

	pt_camera = M_rgb * pt_world;

	pt_image.x = projParams_rgb.x * pt_camera.x / pt_camera.z + projParams_rgb.z;
	pt_image.y = projParams_rgb.y * pt_camera.y / pt_camera.z + projParams_rgb.w;

	if ((pt_image.x < 1) || (pt_image.x > imgSize.x - 2) || (pt_image.y < 1) || (pt_image.y > imgSize.y - 2)) return;

	rgb_measure = TO_VECTOR3(interpolateBilinear(rgb, pt_image, imgSize)) / 255.0f;
	//rgb_measure = rgb[(int)(pt_image.x + 0.5f) + (int)(pt_image.y + 0.5f) * imgSize.x].toVector3().toFloat() / 255.0f;
	newW = 1;

	newC = oldC * oldW + rgb_measure * newW;
	newW = oldW + newW;
	newC /= newW;
	newW = MIN(newW, sceneParams.maxW);

	voxel.clr = TO_UCHAR3(newC * 255.0f);
	voxel.w_color = TVoxel::floatToWeight(newW, sceneParams.maxW);
}

template<bool hasColor, bool hasConfidence, class TVoxel>
struct ComputeUpdatedVoxelInfo;

template<class TVoxel>
struct ComputeUpdatedVoxelInfo<false, false, TVoxel>
{
	_CPU_AND_GPU_CODE_ static void compute(DEVICEPTR(TVoxel)& voxel, const TSDFDirection direction,
	                                       const THREADPTR(Vector4f)& pt_world,
	                                       const CONSTPTR(Matrix4f)& M_d, const CONSTPTR(Vector4f)& projParams_d,
	                                       const CONSTPTR(Matrix4f)& M_rgb, const CONSTPTR(Vector4f)& projParams_rgb,
	                                       const ITMFusionParams& fusionParams,
	                                       const CONSTPTR(ITMSceneParams)& sceneParams,
	                                       const CONSTPTR(float)* depth,
	                                       const CONSTPTR(Vector4f)* depthNormals,
	                                       const CONSTPTR(float)* confidence,
	                                       const CONSTPTR(Vector2i)& imgSize_d,
	                                       const CONSTPTR(Vector4u)* rgb, const CONSTPTR(Vector2i)& imgSize_rgb)
	{
		computeUpdatedVoxelDepthInfo(voxel, direction, pt_world, M_d, projParams_d, fusionParams, sceneParams, depth,
		                             depthNormals, imgSize_d);
	}
};

template<class TVoxel>
struct ComputeUpdatedVoxelInfo<true, false, TVoxel>
{
	_CPU_AND_GPU_CODE_ static void compute(DEVICEPTR(TVoxel)& voxel, const TSDFDirection direction,
	                                       const THREADPTR(Vector4f)& pt_world,
	                                       const THREADPTR(Matrix4f)& M_d, const THREADPTR(Vector4f)& projParams_d,
	                                       const THREADPTR(Matrix4f)& M_rgb, const THREADPTR(Vector4f)& projParams_rgb,
	                                       const ITMFusionParams& fusionParams,
	                                       const CONSTPTR(ITMSceneParams)& sceneParams,
	                                       const CONSTPTR(float)* depth,
	                                       const CONSTPTR(Vector4f)* depthNormals,
	                                       const CONSTPTR(float)* confidence,
	                                       const CONSTPTR(Vector2i)& imgSize_d,
	                                       const CONSTPTR(Vector4u)* rgb, const THREADPTR(Vector2i)& imgSize_rgb)
	{
		float eta = computeUpdatedVoxelDepthInfo(voxel, direction, pt_world, M_d, projParams_d, fusionParams, sceneParams,
		                                         depth, depthNormals, imgSize_d);
		if ((eta > sceneParams.mu) || (fabs(eta / sceneParams.mu) > 0.25f)) return;
		computeUpdatedVoxelColorInfo(voxel, direction, pt_world, M_rgb, projParams_rgb, sceneParams, eta, rgb, imgSize_rgb);
	}
};

template<class TVoxel>
struct ComputeUpdatedVoxelInfo<false, true, TVoxel>
{
	_CPU_AND_GPU_CODE_ static void compute(DEVICEPTR(TVoxel)& voxel, const TSDFDirection direction,
	                                       const THREADPTR(Vector4f)& pt_world,
	                                       const CONSTPTR(Matrix4f)& M_d, const CONSTPTR(Vector4f)& projParams_d,
	                                       const CONSTPTR(Matrix4f)& M_rgb, const CONSTPTR(Vector4f)& projParams_rgb,
	                                       const ITMFusionParams& fusionParams,
	                                       const CONSTPTR(ITMSceneParams)& sceneParams,
	                                       const CONSTPTR(float)* depth,
	                                       const CONSTPTR(Vector4f)* depthNormals,
	                                       const CONSTPTR(float)* confidence,
	                                       const CONSTPTR(Vector2i)& imgSize_d,
	                                       const CONSTPTR(Vector4u)* rgb, const CONSTPTR(Vector2i)& imgSize_rgb)
	{
		computeUpdatedVoxelDepthInfo(voxel, direction, pt_world, M_d, projParams_d, fusionParams, sceneParams, depth,
		                             depthNormals, confidence, imgSize_d);
	}
};

template<class TVoxel>
struct ComputeUpdatedVoxelInfo<true, true, TVoxel>
{
	_CPU_AND_GPU_CODE_ static void compute(DEVICEPTR(TVoxel)& voxel, const TSDFDirection direction,
	                                       const THREADPTR(Vector4f)& pt_world,
	                                       const THREADPTR(Matrix4f)& M_d, const THREADPTR(Vector4f)& projParams_d,
	                                       const THREADPTR(Matrix4f)& M_rgb, const THREADPTR(Vector4f)& projParams_rgb,
	                                       const ITMFusionParams& fusionParams,
	                                       const CONSTPTR(ITMSceneParams)& sceneParams,
	                                       const CONSTPTR(float)* depth,
	                                       const CONSTPTR(Vector4f)* depthNormals,
	                                       const CONSTPTR(float)* confidence,
	                                       const CONSTPTR(Vector2i)& imgSize_d,
	                                       const CONSTPTR(Vector4u)* rgb, const THREADPTR(Vector2i)& imgSize_rgb)
	{
		float eta = computeUpdatedVoxelDepthInfo(voxel, direction, pt_world, M_d, projParams_d, fusionParams, sceneParams,
		                                         depth, depthNormals, confidence,
		                                         imgSize_d);
		if ((eta > sceneParams.mu) || (fabs(eta / sceneParams.mu) > 0.25f)) return;
		computeUpdatedVoxelColorInfo(voxel, direction, pt_world, M_rgb, projParams_rgb, sceneParams, eta, rgb, imgSize_rgb);
	}
};

template<class TVoxel>
_CPU_AND_GPU_CODE_ static void voxelProjectionCarveSpace(DEVICEPTR(TVoxel)& voxel,
                                                         const TSDFDirection direction,
                                                         const Vector4f pt_world,
                                                         const THREADPTR(Matrix4f)& M_d,
                                                         const THREADPTR(Vector4f)& projParams_d,
                                                         const THREADPTR(Matrix4f)& M_rgb,
                                                         const THREADPTR(Vector4f)& projParams_rgb,
                                                         const ITMFusionParams& fusionParams,
                                                         const CONSTPTR(ITMSceneParams)& sceneParams,
                                                         const CONSTPTR(float)* depth,
                                                         const CONSTPTR(Vector4f)* depthNormals,
                                                         const CONSTPTR(float)* confidence,
                                                         const CONSTPTR(Vector2i)& imgSize_d,
                                                         const CONSTPTR(Vector4u)* rgb,
                                                         const THREADPTR(Vector2i)& imgSize_rgb)
{
	Vector2i imgSize = imgSize_d;

	Vector4f pt_camera = M_d * pt_world;
	if (pt_camera.z <= 0) return; // voxel behind camera

	Vector2f voxelCenter_image = project(pt_camera.toVector3(), projParams_d);
	if ((voxelCenter_image.x < 1) || (voxelCenter_image.x > imgSize.x - 2)
	    || (voxelCenter_image.y < 1) || (voxelCenter_image.y > imgSize.y - 2))
		return;

	/// Find relevant image area
	Vector2f voxelBR_image = project(pt_camera.toVector3() + Vector3f(1, 1, 0) * 0.5 * sceneParams.voxelSize,
	                                 projParams_d);
	Vector2f voxelUL_image = project(pt_camera.toVector3() - Vector3f(1, 1, 0) * 0.5 * sceneParams.voxelSize,
	                                 projParams_d);

	int x_min = MAX(static_cast<int>(voxelUL_image.x + 0.5f), 1);
	int x_max = MIN(static_cast<int>(voxelBR_image.x + 0.5f), imgSize.x - 2);
	int y_min = MAX(static_cast<int>(voxelUL_image.y + 0.5f), 1);
	int y_max = MIN(static_cast<int>(voxelBR_image.y + 0.5f), imgSize.y - 2);

	/// Check area and carve voxel, if necessary
	float newF = 1;
	float newW = 0;
	for (int x = x_min; x <= x_max; x++)
		for (int y = y_min; y <= y_max; y++)
		{
			int idx = x + y * imgSize.x;
			// get measured depth from image
			float depthMeasure = depth[idx];
			Vector4f normalCamera = depthNormals[idx];
			if (depthMeasure <= 0.0 or normalCamera.w != 1) continue;
			// check whether voxel needs updating
			float eta = depthMeasure - pt_camera.z;
			if (eta < sceneParams.mu) // Within truncation range -> don't carve
				return;

			newW += 1;
		}
	if (newW <= 0)
		return;

	/// Update SDF
	float oldF = TVoxel::valueToFloat(voxel.sdf);
	float oldW = TVoxel::weightToFloat(voxel.w_depth, sceneParams.maxW);

	newF = oldW * oldF + newW * newF;
	newW = oldW + newW;
	newF /= newW;
	newW = MIN(newW, sceneParams.maxW);

	// write back^
	voxel.sdf = TVoxel::floatToValue(newF);
	voxel.w_depth = TVoxel::floatToWeight(newW, sceneParams.maxW);
}

_CPU_AND_GPU_CODE_
inline void SetBlockVisibleType(const CONSTPTR(ITMHashEntry)* hashTable,
                                DEVICEPTR(Vector4s)* blockCoords,
                                DEVICEPTR(TSDFDirection)* blockDirections,
                                HashEntryVisibilityType* entriesVisibleType,
                                Vector3i blockPos, TSDFDirection direction = TSDFDirection::NONE)
{
	bool useDirectional = (direction != TSDFDirection::NONE);
	//compute index in hash table
	int hashIdx = hashIndex(blockPos, direction);

	//check if hash table contains entry
	bool isFound = false;

	ITMHashEntry hashEntry = hashTable[hashIdx];

	if (IS_EQUAL3(hashEntry.pos, blockPos) and hashEntry.ptr >= -1 and
	    (not useDirectional or hashEntry.direction == static_cast<TSDFDirection_type>(direction)))
	{
		//entry has been streamed out but is visible or in memory and visible
		entriesVisibleType[hashIdx] = (hashEntry.ptr == -1) ? VISIBLE_STREAMED_OUT : VISIBLE_IN_MEMORY;

		isFound = true;
	}

	if (!isFound)
	{
		bool isExcess = false;
		if (hashEntry.ptr >= -1) //seach excess list only if there is no room in ordered part
		{
			while (hashEntry.offset >= 1)
			{
				hashIdx = SDF_BUCKET_NUM + hashEntry.offset - 1;
				hashEntry = hashTable[hashIdx];

				if (IS_EQUAL3(hashEntry.pos, blockPos) and hashEntry.ptr >= -1 and
				    (not useDirectional or hashEntry.direction == static_cast<TSDFDirection_type>(direction)))
				{
					//entry has been streamed out but is visible or in memory and visible
					entriesVisibleType[hashIdx] = (hashEntry.ptr == -1) ? VISIBLE_STREAMED_OUT : VISIBLE_IN_MEMORY;

					isFound = true;
					break;
				}
			}

			isExcess = true;
		}
	}
}

_CPU_AND_GPU_CODE_
inline void SetBlockAllocAndVisibleType(const CONSTPTR(ITMHashEntry)* hashTable,
                                        DEVICEPTR(Vector4s)* blockCoords,
                                        DEVICEPTR(TSDFDirection)* blockDirections,
                                        HashEntryAllocType* entriesAllocType,
                                        HashEntryVisibilityType* entriesVisibleType,
                                        Vector3i blockPos, TSDFDirection direction = TSDFDirection::NONE)
{
	bool useDirectional = (direction != TSDFDirection::NONE);
	//compute index in hash table
	int hashIdx = hashIndex(blockPos, direction);

	//check if hash table contains entry
	bool isFound = false;

	ITMHashEntry hashEntry = hashTable[hashIdx];

	if (IS_EQUAL3(hashEntry.pos, blockPos) and hashEntry.ptr >= -1 and
	    (not useDirectional or hashEntry.direction == static_cast<TSDFDirection_type>(direction)))
	{
		//entry has been streamed out but is visible or in memory and visible
		entriesVisibleType[hashIdx] = (hashEntry.ptr == -1) ? VISIBLE_STREAMED_OUT : VISIBLE_IN_MEMORY;

		isFound = true;
	}

	if (!isFound)
	{
		bool isExcess = false;
		if (hashEntry.ptr >= -1) //seach excess list only if there is no room in ordered part
		{
			while (hashEntry.offset >= 1)
			{
				hashIdx = SDF_BUCKET_NUM + hashEntry.offset - 1;
				hashEntry = hashTable[hashIdx];

				if (IS_EQUAL3(hashEntry.pos, blockPos) and hashEntry.ptr >= -1 and
				    (not useDirectional or hashEntry.direction == static_cast<TSDFDirection_type>(direction)))
				{
					//entry has been streamed out but is visible or in memory and visible
					entriesVisibleType[hashIdx] = (hashEntry.ptr == -1) ? VISIBLE_STREAMED_OUT : VISIBLE_IN_MEMORY;

					isFound = true;
					break;
				}
			}

			isExcess = true;
		}

		if (!isFound) //still not found
		{
			entriesAllocType[hashIdx] = isExcess ? ALLOCATE_EXCESS : ALLOCATE_ORDERED; //needs allocation
			if (!isExcess) entriesVisibleType[hashIdx] = VISIBLE_IN_MEMORY; //new entry is visible

			blockCoords[hashIdx] = Vector4s(blockPos.x, blockPos.y, blockPos.z, 1);
			blockDirections[hashIdx] = direction;
		}
	}
}

template<class TVoxel>
_CPU_AND_GPU_CODE_
void rayCastCarveSpace(int x, int y, Vector2i imgSize, float* depth, Vector4f* depthNormals,
                       const Matrix4f& invM_d,
                       const Vector4f& invProjParams_d, const Vector4f& invProjParams_rgb,
                       const ITMFusionParams& fusionParams,
                       const ITMSceneParams& sceneParams,
                       const ITMHashEntry* hashTable,
                       VoxelRayCastingSum* entriesRayCasting,
                       TVoxel* voxelArray
)
{
	const float mu = sceneParams.mu;
	const float viewFrustum_min = sceneParams.viewFrustum_min;
	const float viewFrustum_max = sceneParams.viewFrustum_max;
	const float voxelSize = sceneParams.voxelSize;

	int idx = y * imgSize.x + x;

	float depthValue = depth[idx];
	Vector4f normal_camera = depthNormals[idx];
	if (depthValue <= 0 or (depthValue - mu) < viewFrustum_min or (depthValue + mu) > viewFrustum_max
	    or normal_camera.w != 1)
		return;


	Vector3f pt_camera = reprojectImagePoint(x, y, depthValue, invProjParams_d);
	Vector3f pt_world = (invM_d * Vector4f(pt_camera, 1)).toVector3();
	Vector4f rayStart_camera = Vector4f(reprojectImagePoint(x, y, viewFrustum_min, invProjParams_d), 1);
	Vector3f rayStart_world = (invM_d * rayStart_camera).toVector3();
	Vector3f rayDirection_world = (pt_world - rayStart_world).normalised();

	float weights[N_DIRECTIONS];
	if (fusionParams.tsdfMode == TSDFMode::TSDFMODE_DIRECTIONAL)
	{
		ComputeDirectionWeights(-rayDirection_world, weights);
	}


	Matrix4f M_d;
	invM_d.inv(M_d);

//	float carveDistance = ORUtils::length(pt_world - rayStart_world) - 1 * mu;
	normal_camera.w = 0; // rotation-only transformation
	Vector3f normal_world = (invM_d * normal_camera).toVector3();
	float carveDistance = ORUtils::length(pt_world - rayStart_world)
	                      - fabs(1.0 / dot(normal_world, rayDirection_world)) * mu;

	BlockTraversal blockTraversal(rayStart_world, rayDirection_world, carveDistance, voxelSize);
	ITMVoxelBlockHash::IndexCache cache[N_DIRECTIONS];
	while (blockTraversal.HasNextBlock())
	{
		Vector3i voxelIdx = blockTraversal.GetNextBlock();

		Vector3i blockPos;
		ushort linearIdx;
		voxelToBlockPosAndOffset(voxelIdx, blockPos, linearIdx);

		/// Fixed values for distance and weight
		float distance = 1;
		// TODO: down weight, if close to surface

		float weight = 1;

		/// find and update voxels
		if (fusionParams.tsdfMode == TSDFMode::TSDFMODE_DIRECTIONAL)
		{
			for (TSDFDirection_type direction = 0; direction < N_DIRECTIONS; direction++)
			{
				int foundEntry = false;
				int index = findVoxel(hashTable, voxelIdx, TSDFDirection(direction), foundEntry, cache[direction]);

				if (not foundEntry)
					continue;

				VoxelRayCastingSum& voxelRayCastingSum = entriesRayCasting[index];
				voxelRayCastingSum.update(distance, weight);
			}
		} else
		{
			int foundEntry = false;
			int index = findVoxel(hashTable, voxelIdx, foundEntry, cache[0]);

			if (not foundEntry)
				continue;

			VoxelRayCastingSum& voxelRayCastingSum = entriesRayCasting[index];
			voxelRayCastingSum.update(distance, weight);
		}
	}
}

template<class TVoxel>
_CPU_AND_GPU_CODE_
void rayCastUpdate(int x, int y, Vector2i imgSize, float* depth, Vector4f* depthNormals,
                   const Matrix4f& invM_d,
                   const Vector4f& invProjParams_d, const Vector4f& invProjParams_rgb,
                   const ITMFusionParams& fusionParams,
                   const ITMSceneParams& sceneParams,
                   const ITMHashEntry* hashTable,
                   VoxelRayCastingSum* entriesRayCasting
)
{
	const float mu = sceneParams.mu;
	const float viewFrustum_min = sceneParams.viewFrustum_min;
	const float viewFrustum_max = sceneParams.viewFrustum_max;
	const float voxelSize = sceneParams.voxelSize;

	int idx = y * imgSize.x + x;

	float depthValue = depth[idx];
	Vector4f normal_camera = depthNormals[idx];

	if (depthValue <= 0 or (depthValue - mu) < viewFrustum_min or
	    (depthValue + mu) > viewFrustum_max or normal_camera.w != 1)
		return;

	normal_camera.w = 0; // rotation-only transformation
	Vector4f pt_camera = Vector4f(reprojectImagePoint(x, y, depthValue, invProjParams_d), 1);
	Vector3f pt_world = (invM_d * pt_camera).toVector3();
	Vector3f normal_world = (invM_d * normal_camera).toVector3();

	Vector3f viewRay_camera = reprojectImagePoint(x, y, 1, invProjParams_d).normalised();

	float weights[N_DIRECTIONS];
	if (fusionParams.tsdfMode == TSDFMode::TSDFMODE_DIRECTIONAL)
	{
		ComputeDirectionWeights(normal_world, weights);
	}

	Vector3f rayDirectionBefore, rayDirectionBehind;

	if (fusionParams.fusionMode == FusionMode::FUSIONMODE_RAY_CASTING_VIEW_DIR_AND_NORMAL)
	{
		rayDirectionBefore = -(invM_d * Vector4f(pt_camera.toVector3().normalised(), 0)).toVector3();
		rayDirectionBehind = -normal_world;
	} else if (fusionParams.fusionMode == FusionMode::FUSIONMODE_RAY_CASTING_VIEW_DIR)
	{
		rayDirectionBefore = -(invM_d * Vector4f(pt_camera.toVector3().normalised(), 0)).toVector3();
		rayDirectionBehind = (invM_d * Vector4f(pt_camera.toVector3().normalised(), 0)).toVector3();
	} else
	{
		rayDirectionBefore = normal_world;
		rayDirectionBehind = -normal_world;
	}

	BlockTraversal blockTraversalBefore(pt_world, rayDirectionBefore, mu, voxelSize);
	BlockTraversal blockTraversalBehind(pt_world, rayDirectionBehind, mu, voxelSize);
	if (blockTraversalBehind.HasNextBlock()) blockTraversalBehind.GetNextBlock(); // Skip first voxel to prevent duplicate fusion

	ITMVoxelBlockHash::IndexCache cache[N_DIRECTIONS];
	while (blockTraversalBefore.HasNextBlock() or blockTraversalBehind.HasNextBlock())
	{
		Vector3i voxelIdx;
		if (blockTraversalBefore.HasNextBlock())
			voxelIdx = blockTraversalBefore.GetNextBlock();
		else
			voxelIdx = blockTraversalBehind.GetNextBlock();

		Vector3f voxelPos = blockTraversalBefore.BlockToWorld(voxelIdx);

		/// compute distance
		float distance;
		Vector3f voxelSurfaceOffset = voxelPos - pt_world;
		if (fusionParams.fusionMetric == FusionMetric::FUSIONMETRIC_POINT_TO_PLANE)
		{
			distance = ORUtils::dot(voxelSurfaceOffset, normal_world);
		} else
		{
			distance = SIGN(ORUtils::dot(voxelSurfaceOffset, normal_world)) * ORUtils::length(voxelSurfaceOffset);
		}
		distance = MAX(-1.0, MIN(1.0f, distance / mu));

		float weight = 1;
		float voxelSideLengthCamera = voxelSize / (invProjParams_d.x * depthValue); // in pixels

		/// find and update voxels
		Vector3i blockPos;
		ushort linearIdx;
		voxelToBlockPosAndOffset(voxelIdx, blockPos, linearIdx);
		if (fusionParams.tsdfMode == TSDFMode::TSDFMODE_DIRECTIONAL)
		{
			for (TSDFDirection_type direction = 0; direction < N_DIRECTIONS; direction++)
			{
				if (weights[direction] < direction_weight_threshold)
					continue;

				int foundEntry = false;
				int index = findVoxel(hashTable, voxelIdx, TSDFDirection(direction), foundEntry, cache[direction]);
				if (not foundEntry)
				{
					break;
				}

				if (fusionParams.useWeighting)
				{
					float directionWeight = DirectionWeight(normal_world, TSDFDirection(direction));
					float voxelDistanceWeight = 1 - MIN(length(voxelSurfaceOffset) / mu, 1);

					weight = depthWeight(depthValue, normal_camera.toVector3(), viewRay_camera, directionWeight, sceneParams)
					         / powf(voxelSize * 100, 3) * voxelDistanceWeight;

					// Normalize by voxel size in camera image (max num rays to hit), so comparable to voxelProjection fusion
					weight *= 1 / (voxelSideLengthCamera * voxelSideLengthCamera);
				}

				VoxelRayCastingSum& voxelRayCastingSum = entriesRayCasting[index];
				voxelRayCastingSum.update(distance, weight);
			}
		} else
		{
			int foundEntry = false;
			int index = findVoxel(hashTable, voxelIdx, foundEntry, cache[0]);
			if (not foundEntry)
			{
				break;
			}

			if (fusionParams.useWeighting)
			{
				float voxelDistanceWeight = 1 - MIN(length(voxelSurfaceOffset) / mu, 1);
				weight = depthWeight(depthValue, normal_camera.toVector3(), viewRay_camera, 1, sceneParams)
				         / powf(voxelSize * 100, 3) * voxelDistanceWeight;

				// Normalize by voxel size in camera image (max num rays to hit), so comparable to voxelProjection fusion
				weight *= 1 / (voxelSideLengthCamera * voxelSideLengthCamera);
			}

			VoxelRayCastingSum& voxelRayCastingSum = entriesRayCasting[index];
			voxelRayCastingSum.update(distance, weight);
		}
	}
}

/**
 * Collect and combine summed voxels after ray cast update.
 * @tparam TVoxel
 */
template<class TVoxel>
_CPU_AND_GPU_CODE_
void rayCastCombine(TVoxel& voxel, const VoxelRayCastingSum& rayCastingSum, const ITMSceneParams& sceneParams)
{
	float deltaSDF = rayCastingSum.sdfSum / rayCastingSum.weightSum;
	float deltaWeight = rayCastingSum.weightSum;
	if (deltaWeight == 0)
		return;

	float currentSDF = TVoxel::valueToFloat(voxel.sdf);
	float currentWeight = TVoxel::weightToFloat(voxel.w_depth, sceneParams.maxW);

	if (sceneParams.stopIntegratingAtMaxW and currentWeight == sceneParams.maxW)
		return;

	float newWeight = currentWeight + deltaWeight;
	float newSDF = (currentWeight * currentSDF + deltaWeight * deltaSDF) / newWeight;
	newWeight = MIN(newWeight, sceneParams.maxW);

	voxel.sdf = TVoxel::floatToValue(newSDF);
	voxel.w_depth = TVoxel::floatToWeight(newWeight, sceneParams.maxW);
}

/**
 * Ray cast depth image to find visible blocks for space carving
 *
 * @param entriesAllocType Per HashEntry indicator whether it requires allocation
 * @param entriesVisibleType Per HashEntry indicator if block is visible
 * @param x
 * @param y
 * @param blockCoords
 * @param depth
 * @param invM_d
 * @param projParams_d
 * @param mu
 * @param imgSize
 * @param voxelSize
 * @param hashTable
 * @param viewFrustum_min
 * @param viewFrustum_max
 */
_CPU_AND_GPU_CODE_ inline void
buildSpaceCarvingVisibleType(DEVICEPTR(HashEntryVisibilityType)* entriesVisibleType,
                             int x, int y,
                             DEVICEPTR(Vector4s)* blockCoords, DEVICEPTR(TSDFDirection)* blockDirections,
                             const CONSTPTR(float)* depth,
                             const CONSTPTR(Vector4f)* depthNormal, Matrix4f invM_d,
                             Vector4f projParams_d, float mu, Vector2i imgSize, float voxelSize,
                             const CONSTPTR(ITMHashEntry)* hashTable, float viewFrustum_min, float viewFrustum_max,
                             const ITMFusionParams& fusionParams
)
{
	float depth_measure = depth[x + y * imgSize.x];
	if (depth_measure <= 0 or (depth_measure - mu) < 0 or (depth_measure - mu) < viewFrustum_min or
	    (depth_measure + mu) > viewFrustum_max)
		return;

	Vector3f pt_camera = reprojectImagePoint(x, y, depth_measure, projParams_d);
	Vector3f pt_world = (invM_d * Vector4f(pt_camera, 1)).toVector3();
	Vector4f rayStart_camera = Vector4f(reprojectImagePoint(x, y, viewFrustum_min, projParams_d), 1);
	Vector3f rayStart_world = (invM_d * rayStart_camera).toVector3();
	Vector3f rayDirection_world = (pt_world - rayStart_world).normalised();

//	float carveDistance = ORUtils::length(pt_world - rayStart_world)
//	                      - fabs(1.0 / dot(normal_world, rayDirection_world)) * mu;
	float carveDistance = ORUtils::length(pt_world - rayStart_world) - 1 * mu;
	BlockTraversal blockTraversal_carving(rayStart_world, rayDirection_world, carveDistance,
	                                      voxelSize * SDF_BLOCK_SIZE, false);
	while (blockTraversal_carving.HasNextBlock())
	{
		Vector3i blockPos = blockTraversal_carving.GetNextBlock();

		if (fusionParams.tsdfMode == TSDFMode::TSDFMODE_DIRECTIONAL)
		{
			for (TSDFDirection_type direction = 0; direction < N_DIRECTIONS; direction++)
			{
				SetBlockVisibleType(hashTable, blockCoords, blockDirections, entriesVisibleType,
				                    blockPos, TSDFDirection(direction));
			}
		} else
		{
			SetBlockVisibleType(hashTable, blockCoords, blockDirections, entriesVisibleType, blockPos);
		}
	}
}

/**
 * Ray cast depth image to find visible blocks and determine whether they need allocation.
 *
 * @param entriesAllocType Per HashEntry indicator whether it requires allocation
 * @param entriesVisibleType Per HashEntry indicator if block is visible
 * @param x
 * @param y
 * @param blockCoords
 * @param depth
 * @param invM_d
 * @param projParams_d
 * @param mu
 * @param imgSize
 * @param voxelSize
 * @param hashTable
 * @param viewFrustum_min
 * @param viewFrustum_max
 */
_CPU_AND_GPU_CODE_ inline void
buildHashAllocAndVisibleType(DEVICEPTR(HashEntryAllocType)* entriesAllocType,
                             DEVICEPTR(HashEntryVisibilityType)* entriesVisibleType,
                             int x, int y,
                             DEVICEPTR(Vector4s)* blockCoords, DEVICEPTR(TSDFDirection)* blockDirections,
                             const CONSTPTR(float)* depth,
                             const CONSTPTR(Vector4f)* depthNormal, Matrix4f invM_d,
                             Vector4f projParams_d, float mu, Vector2i imgSize, float voxelSize,
                             const CONSTPTR(ITMHashEntry)* hashTable, float viewFrustum_min, float viewFrustum_max,
                             const ITMFusionParams& fusionParams
)
{
	float depth_measure = depth[x + y * imgSize.x];
	Vector4f normal_camera = depthNormal[x + y * imgSize.x];
	if (depth_measure <= 0 or (depth_measure - mu) < 0 or (depth_measure - mu) < viewFrustum_min or
	    (depth_measure + mu) > viewFrustum_max or normal_camera.w != 1)
		return;

	normal_camera.w = 0; // rotation-only transformation
	Vector3f normalWorld = (invM_d * normal_camera).toVector3();

	Vector4f pt_camera = Vector4f(reprojectImagePoint(x, y, depth_measure, projParams_d), 1);

	Vector3f pt_world = (invM_d * pt_camera).toVector3();

	Vector3f rayDirectionBefore, rayDirectionBehind;
	if (fusionParams.fusionMode == FusionMode::FUSIONMODE_VOXEL_PROJECTION
	    or fusionParams.fusionMode == FusionMode::FUSIONMODE_RAY_CASTING_VIEW_DIR)
	{
		Vector4f camera_ray_world = invM_d * Vector4f(pt_camera.toVector3().normalised(), 0);
		rayDirectionBefore = -camera_ray_world.toVector3();
		rayDirectionBehind = camera_ray_world.toVector3();
	} else
	{

		rayDirectionBehind = -normalWorld;

		if (fusionParams.fusionMode == FusionMode::FUSIONMODE_RAY_CASTING_VIEW_DIR_AND_NORMAL)
			rayDirectionBefore = -(invM_d * Vector4f(pt_camera.toVector3().normalised(), 0)).toVector3();
		else
			rayDirectionBefore = normalWorld;
	}

	BlockTraversal blockTraversalBefore(pt_world, rayDirectionBefore, mu, voxelSize);
	BlockTraversal blockTraversalBehind(pt_world, rayDirectionBehind, mu, voxelSize);
	if (blockTraversalBehind.HasNextBlock()) blockTraversalBehind.GetNextBlock(); // Skip first voxel to prevent duplicate fusion

	float weights[N_DIRECTIONS];
	ComputeDirectionWeights(normalWorld, weights);

	Vector3i lastBlockPos(MAX_INT, MAX_INT, MAX_INT);
	while (blockTraversalBefore.HasNextBlock() or blockTraversalBehind.HasNextBlock())
	{
		Vector3i voxelPos;
		if (blockTraversalBefore.HasNextBlock())
			voxelPos = blockTraversalBefore.GetNextBlock();
		else
			voxelPos = blockTraversalBehind.GetNextBlock();

		Vector3i blockPos = voxelToBlockPos(voxelPos);
		if (blockPos == lastBlockPos)
			continue;

		if (fusionParams.tsdfMode == TSDFMode::TSDFMODE_DIRECTIONAL)
		{
			for (TSDFDirection_type direction = 0; direction < N_DIRECTIONS; direction++)
			{
				if (weights[direction] < direction_weight_threshold)
					continue;
				SetBlockAllocAndVisibleType(hashTable, blockCoords, blockDirections, entriesAllocType, entriesVisibleType,
				                            blockPos, TSDFDirection(direction));
			}
		} else
		{
			SetBlockAllocAndVisibleType(hashTable, blockCoords, blockDirections, entriesAllocType, entriesVisibleType,
			                            blockPos);
		}
		lastBlockPos = blockPos;
	}
}


template<bool useSwapping>
_CPU_AND_GPU_CODE_ inline void checkPointVisibility(THREADPTR(bool)& isVisible, THREADPTR(bool)& isVisibleEnlarged,
                                                    const THREADPTR(Vector4f)& pt_image, const CONSTPTR(Matrix4f)& M_d,
                                                    const CONSTPTR(Vector4f)& projParams_d,
                                                    const CONSTPTR(Vector2i)& imgSize)
{
	Vector4f pt_buff;

	pt_buff = M_d * pt_image;

	if (pt_buff.z < 1e-10f) return;

	pt_buff.x = projParams_d.x * pt_buff.x / pt_buff.z + projParams_d.z;
	pt_buff.y = projParams_d.y * pt_buff.y / pt_buff.z + projParams_d.w;

	if (pt_buff.x >= 0 && pt_buff.x < imgSize.x && pt_buff.y >= 0 && pt_buff.y < imgSize.y)
	{
		isVisible = true;
		isVisibleEnlarged = true;
	} else if (useSwapping)
	{
		Vector4i lims;
		lims.x = -imgSize.x / 8;
		lims.y = imgSize.x + imgSize.x / 8;
		lims.z = -imgSize.y / 8;
		lims.w = imgSize.y + imgSize.y / 8;

		if (pt_buff.x >= lims.x && pt_buff.x < lims.y && pt_buff.y >= lims.z && pt_buff.y < lims.w)
			isVisibleEnlarged = true;
	}
}

template<bool useSwapping>
_CPU_AND_GPU_CODE_ inline void checkBlockVisibility(THREADPTR(bool)& isVisible, THREADPTR(bool)& isVisibleEnlarged,
                                                    const THREADPTR(Vector3s)& hashPos, const CONSTPTR(Matrix4f)& M_d,
                                                    const CONSTPTR(Vector4f)& projParams_d,
                                                    const CONSTPTR(float)& voxelSize, const CONSTPTR(Vector2i)& imgSize)
{
	Vector4f pt_image;
	float factor = (float) SDF_BLOCK_SIZE * voxelSize;

	isVisible = false;
	isVisibleEnlarged = false;

	// 0 0 0
	pt_image.x = (float) hashPos.x * factor;
	pt_image.y = (float) hashPos.y * factor;
	pt_image.z = (float) hashPos.z * factor;
	pt_image.w = 1.0f;
	checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;

	// 0 0 1
	pt_image.z += factor;
	checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;

	// 0 1 1
	pt_image.y += factor;
	checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;

	// 1 1 1
	pt_image.x += factor;
	checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;

	// 1 1 0 
	pt_image.z -= factor;
	checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;

	// 1 0 0 
	pt_image.y -= factor;
	checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;

	// 0 1 0
	pt_image.x -= factor;
	pt_image.y += factor;
	checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;

	// 1 0 1
	pt_image.x += factor;
	pt_image.y -= factor;
	pt_image.z += factor;
	checkPointVisibility<useSwapping>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;
}

template<bool useSwapping>
_CPU_AND_GPU_CODE_
void buildVisibleList(ITMHashEntry* hashTable, ITMHashSwapState* swapStates, int noTotalEntries,
                      int* visibleEntryIDs, AllocationTempData* allocData, HashEntryVisibilityType* entriesVisibleType,
                      Matrix4f M_d, Vector4f projParams_d, Vector2i depthImgSize, float voxelSize,
                      int targetIdx)
{
	HashEntryVisibilityType hashVisibleType = entriesVisibleType[targetIdx];
	const ITMHashEntry& hashEntry = hashTable[targetIdx];

	if (hashVisibleType == PREVIOUSLY_VISIBLE)
	{
		bool isVisibleEnlarged, isVisible;

		if (useSwapping)
		{
			checkBlockVisibility<true>(isVisible, isVisibleEnlarged, hashEntry.pos, M_d, projParams_d, voxelSize,
			                           depthImgSize);
			if (!isVisibleEnlarged) hashVisibleType = INVISIBLE;
		} else
		{
			checkBlockVisibility<false>(isVisible, isVisibleEnlarged, hashEntry.pos, M_d, projParams_d, voxelSize,
			                            depthImgSize);
			if (!isVisible) hashVisibleType = INVISIBLE;
		}
		entriesVisibleType[targetIdx] = hashVisibleType;
	}

	if (useSwapping)
	{
		if (hashVisibleType > 0 && swapStates[targetIdx].state != 2) swapStates[targetIdx].state = 1;
	}
}

} // namespace ITMLib
