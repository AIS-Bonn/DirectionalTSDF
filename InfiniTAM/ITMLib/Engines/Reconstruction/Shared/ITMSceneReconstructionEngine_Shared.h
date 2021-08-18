// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <type_traits>
#include "Objects/Scene/ITMRepresentationAccess.h"
#include "Objects/Scene/ITMDirectional.h"
#include "Utils/ITMPixelUtils.h"
#include "ITMBlockTraversal.h"
#include "ITMLib/Engines/Reconstruction/Interface/ITMSceneReconstructionEngine.h"
#include "ITMLib/Engines/Reconstruction/Shared/ITMFusionWeight.hpp"
#include "ITMLib/Engines/Reconstruction/Shared/ITMSummingVoxelMap.h"
#include "ITMLib/Utils/ITMProjectionUtils.h"
#include "ITMLib/Objects/Scene/ITMSummingVoxel.h"

namespace ITMLib
{

_CPU_AND_GPU_CODE_ inline Vector3f
normalCameraToWorld(const Vector4f& normal_camera, const Matrix4f& invM)
{
	return (invM * Vector4f(normal_camera.toVector3(), 0)).toVector3();
}

template<typename TVoxel>
_CPU_AND_GPU_CODE_ inline float
computeUpdatedVoxelDepthInfo(ITMVoxel& voxel, const TSDFDirection direction,
                             const Vector4f& voxel_world,
                             const Matrix4f& M_d,
                             const Vector4f& projParams_d,
                             const ITMFusionParams& fusionParams,
                             const ITMSceneParams& sceneParams,
                             const float* depth,
                             const Vector4f* depthNormals,
                             const Vector2i& imgSize
)
{
	Vector4f voxel_camera;
	Vector2f voxel_image;
	float depth_measure, oldF, newF;
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
	Vector4f normal_camera = depthNormals[idx];
	if (depth_measure <= 0.0f or normal_camera.w != 1) return -1;

	// check whether voxel needs updating
	float voxelSurfaceOffset = depth_measure - voxel_camera.z;
	if (voxelSurfaceOffset < -sceneParams.mu) return voxelSurfaceOffset;

	Matrix4f invM_d;
	M_d.inv(invM_d);
	Vector3f pt_camera = reprojectImagePoint(voxel_image.x, voxel_image.y, depth_measure,
	                                         invertProjectionParams(projParams_d));
	Vector3f pt_world = (invM_d * Vector4f(pt_camera, 1)).toVector3();
	Vector3f normal_world = normalCameraToWorld(normal_camera, invM_d);
	Vector3f pixelRay_world = (invM_d * Vector4f(pt_camera, 0)).toVector3().normalised();

	float distance;
	if (fusionParams.fusionMetric == FusionMetric::FUSIONMETRIC_POINT_TO_PLANE)
	{
		distance = ORUtils::dot(voxel_world.toVector3() - pt_world, normal_world);
	} else
	{
		// True point-to-point (euclidean distance)
		distance = ORUtils::dot(voxel_world.toVector3() - pt_world, -pixelRay_world);

		// Original InfiniTAM: assumption is, all surface normals equal the inverse camera Z-axis
//		Vector3f cameraRay_world = (invM_d * Vector4f(0, 0, -1, 0)).toVector3();
//		distance = ORUtils::dot(voxel_world.toVector3() - pt_world, cameraRay_world);
	}

	// compute updated SDF value and reliability
	oldF = ITMVoxel::valueToFloat(voxel.sdf);
	oldW = ITMVoxel::weightToFloat(voxel.w_depth, sceneParams.maxW);
	newF = MAX(MIN(1.0f, distance / sceneParams.mu), -1.0f);
	newW = 1;

	if (fusionParams.useWeighting) //and distance < carveSurfaceOffset)
	{
		if (distance >= sceneParams.mu or ORUtils::length(voxel_world.toVector3() - pt_world) > 2 * sceneParams.mu)
		{ // space carving region (optional)
			Vector3f viewRay_camera = reprojectImagePoint(voxel_image.x, voxel_image.y, 1,
			                                              invertProjectionParams(projParams_d)).normalised();
			newF = 1.0f;
			newW = 0.01f * weightNormal(normal_camera.toVector3(), viewRay_camera);
		} else
		{ // truncation region
			Vector3f viewRay_camera = reprojectImagePoint(voxel_image.x, voxel_image.y, 1,
			                                              invertProjectionParams(projParams_d)).normalised();
			if (direction != TSDFDirection::NONE)
			{
				float directionAngle = DirectionAngle(normal_world, direction);
				float directionWeight = DirectionWeight(directionAngle);
				newW = combinedWeight(depth_measure, distance, normal_camera.toVector3(), viewRay_camera, sceneParams) * directionWeight;
//				newW *= (ORUtils::dot(normal_camera.toVector3(), -viewRay_camera) > 0.3); // don't fuse too steep angles in directional, as unreliable
//				if (distance > 0 and directionAngle > direction_angle_threshold and directionAngle < M_PI_2)
//					newW = 0.01f * weightNormal(normal_camera.toVector3(), viewRay_camera);
			} else
			{
				newW = combinedWeight(depth_measure, distance, normal_camera.toVector3(), viewRay_camera, sceneParams);
			}
		}
	}
	if (newW <= 0)
		return -1;

//	{
//		Vector3i voxelIdx = worldPosToVoxelIdx(voxel_world.toVector3(), sceneParams.voxelSize);
//		if (voxelIdx.x == -4 and voxelIdx.y == -2 and voxelIdx.z == 12)
//			printf("################%i (%f, %f)\n", direction, newF, newW);
//	}

	newF = oldW * oldF + newW * newF;
	newW = oldW + newW;
	newF /= newW;
	newW = MIN(newW, sceneParams.maxW);

	// write back
	voxel.sdf = ITMVoxel::floatToValue(newF);
	voxel.w_depth = ITMVoxel::floatToWeight(newW, sceneParams.maxW);

	return distance;
}

template<typename TVoxel>
_CPU_AND_GPU_CODE_ inline void
computeUpdatedVoxelColorInfo(TVoxel& voxel, const TSDFDirection direction,
                             const Vector4f& voxel_world,
                             const Matrix4f& M_d, const Vector4f& projParams_d,
                             const Matrix4f& M_rgb, const Vector4f& projParams_rgb,
                             const ITMFusionParams& fusionParams,
                             const ITMSceneParams& sceneParams,
                             const float* depth, const Vector4f* depthNormals, const Vector4u* rgb, const Vector2i& imgSize_d, const Vector2i& imgSize_rgb)
{
	// project point into image
	Vector4f voxel_rgb = M_rgb * voxel_world;
	Vector4f voxel_d = M_d * voxel_world;
	if (voxel_rgb.z <= 0 or voxel_d.z <= 0) return;

	Vector2f voxel_image_d = project(voxel_d.toVector3(), projParams_d);
	if ((voxel_image_d.x < 1) || (voxel_image_d.x > imgSize_d.width - 2) || (voxel_image_d.y < 1) || (voxel_image_d.y > imgSize_d.height - 2))
		return;

	// get depth and normal
	Vector2i imgCoords_d((int) (voxel_image_d.x + 0.5f), (int) (voxel_image_d.y + 0.5f));
	int idx_d = imgCoords_d.x + imgCoords_d.y * imgSize_d.width;
	float depth_measure = depth[idx_d];
	Vector4f normal_camera = depthNormals[idx_d];
	if (depth_measure <= 0.0f or normal_camera.w != 1) return;

	// get color
	float colorWeight_new = 1;
	Vector3f color_new;
	Vector2f voxel_image_rgb = project(voxel_rgb.toVector3(), projParams_rgb);
	if ((voxel_image_rgb.x < 1) || (voxel_image_rgb.x > imgSize_rgb.width - 2) || (voxel_image_rgb.y < 1) || (voxel_image_rgb.y > imgSize_rgb.height - 2))
	{
		colorWeight_new = 0;
	}
	else
	{
		color_new = rgb[(int)(voxel_image_rgb.x + 0.5f) + (int)(voxel_image_rgb.y + 0.5f) * imgSize_rgb.width].toVector3().toFloat() / 255.0f;
//		color_new = TO_VECTOR3(interpolateBilinear(rgb, voxel_image_rgb, imgSize_rgb)) / 255.0f;
	}

	Matrix4f invM_d;
	M_d.inv(invM_d);
	Vector3f pt_camera = reprojectImagePoint((int) (voxel_image_d.x + 0.5f), (int) (voxel_image_d.y + 0.5f),
	                                         depth_measure, invertProjectionParams(projParams_d));
	Vector3f pt_world = (invM_d * Vector4f(pt_camera, 1)).toVector3();
	Vector3f normal_world = normalCameraToWorld(normal_camera, invM_d);
	Vector3f pixelRay_world = (invM_d * Vector4f(voxel_d.toVector3(), 0)).toVector3().normalised();

	float distance;
	if (fusionParams.fusionMetric == FusionMetric::FUSIONMETRIC_POINT_TO_PLANE)
	{
		distance = ORUtils::dot(voxel_world.toVector3() - pt_world, normal_world);
	} else
	{
		// True point-to-point (euclidean distance)
		distance = ORUtils::dot(voxel_world.toVector3() - pt_world, -pixelRay_world);

		// Original InfiniTAM: assumption is, all surface normals equal the inverse camera Z-axis
//		Vector3f cameraRay_world = (invM_d * Vector4f(0, 0, -1, 0)).toVector3();
//		distance = ORUtils::dot(voxel_world.toVector3() - pt_world, cameraRay_world);
	}
	if (distance < -sceneParams.mu) return;

	Vector3f viewRay_camera = reprojectImagePoint(voxel_image_d.x, voxel_image_d.y, 1,
	                                              invertProjectionParams(projParams_d)).normalised();

	float sdf_old = ITMVoxel::valueToFloat(voxel.sdf);
	float sdf_new = MAX(MIN(1.0f, distance / sceneParams.mu), -1.0f);
	float depthWeight_new = 1;
	if (fusionParams.useWeighting) //and distance < carveSurfaceOffset)
	{
		if ((distance >= sceneParams.mu or ORUtils::length(voxel_world.toVector3() - pt_world) > 2 * sceneParams.mu)
//			and false
			)
		{ // space carving range (optional)
//			if (sdf_old > 0) // only apply to negative, as positive might be used for estimating surface -> DEFINITELY WORSE
//				depthWeight_new = 0.01;

			depthWeight_new *= weightDepth(1, sceneParams) * weightNormal(normal_camera.toVector3(), viewRay_camera); // use constant depth weigh, because outside truncation range

			if (direction != TSDFDirection::NONE)
			{
//				depthWeight_new *= static_cast<float>(DirectionAngle(-pixelRay_world, direction) < direction_angle_threshold);
//				depthWeight_new *= static_cast<float>(DirectionAngle(-pixelRay_world, direction) < 2 * M_PI_4);

//				depthWeight_new *=  DirectionWeight(DirectionAngle(-pixelRay_world, direction));
				depthWeight_new *=  (M_PI_2 - DirectionAngle(-pixelRay_world, direction)) / M_PI_2;
			}

			// Check if pixel near to depth gap (large offset to neighboring pixels) in which case, don't carve (retain edges of objects)
			float voxelSizeImage = sceneParams.voxelSize / voxel_d.z * projParams_d.x;
			bool dontCarve = false;
			for (int i = 1; i <= 2 and not dontCarve; i++)
//			for (int i = 1; i < ceil(voxelSizeImage / 2) and not dontCarve; i++)
			{
				dontCarve |= fabs(depth_measure - depth[imgCoords_d.x + i + (imgCoords_d.y + 0) * imgSize_d.width]) > sceneParams.mu;
				dontCarve |= fabs(depth_measure - depth[imgCoords_d.x - i + (imgCoords_d.y + 0) * imgSize_d.width]) > sceneParams.mu;
				dontCarve |= fabs(depth_measure - depth[imgCoords_d.x + 0 + (imgCoords_d.y + i) * imgSize_d.width]) > sceneParams.mu;
				dontCarve |= fabs(depth_measure - depth[imgCoords_d.x + 0 + (imgCoords_d.y - i) * imgSize_d.width]) > sceneParams.mu;

				dontCarve |= depth[imgCoords_d.x + i + (imgCoords_d.y + 0) * imgSize_d.width] <= 0;
				dontCarve |= depth[imgCoords_d.x - i + (imgCoords_d.y + 0) * imgSize_d.width] <= 0;
				dontCarve |= depth[imgCoords_d.x + 0 + (imgCoords_d.y + i) * imgSize_d.width] <= 0;
				dontCarve |= depth[imgCoords_d.x + 0 + (imgCoords_d.y - i) * imgSize_d.width] <= 0;
			}
			if (dontCarve)
				depthWeight_new *= 1e-3; // use really small weight, so errors are carved out eventually

			colorWeight_new = 0;
		} else
		{ // inside truncation range
			depthWeight_new = combinedWeight(depth_measure, distance, normal_camera.toVector3(), viewRay_camera, sceneParams);
			if (direction != TSDFDirection::NONE)
			{
				float directionAngle = DirectionAngle(normal_world, direction);
				depthWeight_new *= DirectionWeight(directionAngle);

				// Fuse positive values, even if angle larger than threshold for direction (carve freespace)
//				if (distance > 0 and directionAngle > direction_angle_threshold and DirectionAngle(-pixelRay_world, direction) < 2 * M_PI_4) // too much? problematic with thin objects
				if (distance > 0 and directionAngle > direction_angle_threshold and DirectionAngle(-pixelRay_world, direction) < direction_angle_threshold)
					depthWeight_new = combinedWeight(depth_measure, distance, normal_camera.toVector3(), viewRay_camera, sceneParams);
//					depthWeight_new = weightDepth(1, sceneParams) * weightNormal(normal_camera.toVector3(), viewRay_camera);
			}
			colorWeight_new *= MAX(0, depthWeight_new * (1 - MIN(1.0f, ORUtils::length(voxel_world.toVector3() - pt_world) / sceneParams.mu))); // distance from surface
		}
	}
	if (depthWeight_new <= 0)
		return;

	// Compute updated values
	float depthWeight_old = ITMVoxel::weightToFloat(voxel.w_depth, sceneParams.maxW);
	float colorWeight_old = ITMVoxel::weightToFloat(voxel.w_color, sceneParams.maxW);
	sdf_new = depthWeight_old * sdf_old + depthWeight_new * sdf_new;
	depthWeight_new = depthWeight_old + depthWeight_new;
	sdf_new /= depthWeight_new;
	depthWeight_new = MIN(depthWeight_new, sceneParams.maxW);

	Vector3f color_old = TO_FLOAT3(voxel.clr) / 255.0f;
	color_new = color_old * colorWeight_old + color_new * colorWeight_new;
	colorWeight_new = colorWeight_new + colorWeight_old;
	color_new /= colorWeight_new;
	colorWeight_new = MIN(colorWeight_new, sceneParams.maxW);

	// write back
	voxel.sdf = ITMVoxel::floatToValue(sdf_new);
	voxel.w_depth = ITMVoxel::floatToWeight(depthWeight_new, sceneParams.maxW);
	voxel.clr = TO_UCHAR3(color_new * 255.0f);
	voxel.w_color = ITMVoxel::floatToWeight(colorWeight_new, sceneParams.maxW);
}

template<bool hasColor, typename TVoxel> // Templating, to prevent building color update parts, if ITMVoxel doesn't contain color
struct ComputeUpdatedVoxelInfo;

template<typename TVoxel>
struct ComputeUpdatedVoxelInfo<false, TVoxel>
{
	_CPU_AND_GPU_CODE_ static void compute(TVoxel& voxel, const TSDFDirection direction,
	                                       const Vector4f& pt_world,
	                                       const Matrix4f& M_d, const Vector4f& projParams_d,
	                                       const Matrix4f& M_rgb, const Vector4f& projParams_rgb,
	                                       const ITMFusionParams& fusionParams,
	                                       const ITMSceneParams& sceneParams,
	                                       const float* depth,
	                                       const Vector4f* depthNormals,
	                                       const float* confidence,
	                                       const Vector2i& imgSize_d,
	                                       const Vector4u* rgb, const Vector2i& imgSize_rgb)
	{
		computeUpdatedVoxelDepthInfo<TVoxel>(voxel, direction, pt_world, M_d, projParams_d, fusionParams, sceneParams,
		                                     depth,
		                                     depthNormals, imgSize_d);
	}
};

template<typename TVoxel>
struct ComputeUpdatedVoxelInfo<true, TVoxel>
{
	_CPU_AND_GPU_CODE_ static void compute(TVoxel& voxel, const TSDFDirection direction,
	                                       const Vector4f& pt_world,
	                                       const Matrix4f& M_d, const Vector4f& projParams_d,
	                                       const Matrix4f& M_rgb, const Vector4f& projParams_rgb,
	                                       const ITMFusionParams& fusionParams,
	                                       const ITMSceneParams& sceneParams,
	                                       const float* depth,
	                                       const Vector4f* depthNormals,
	                                       const float* confidence,
	                                       const Vector2i& imgSize_d,
	                                       const Vector4u* rgb, const Vector2i& imgSize_rgb)
	{
		computeUpdatedVoxelColorInfo<TVoxel>(voxel, direction, pt_world, M_d, projParams_d, M_rgb, projParams_rgb, fusionParams,
		                                     sceneParams, depth, depthNormals, rgb, imgSize_d, imgSize_rgb);
	}
};

_CPU_AND_GPU_CODE_ static void voxelProjectionCarveSpace(const ITMVoxel& voxel,
                                                         SummingVoxel& voxelRayCastingSum,
                                                         const TSDFDirection direction,
                                                         const Vector4f pt_world,
                                                         const Matrix4f& M_d,
                                                         const Vector4f& projParams_d,
                                                         const Matrix4f& M_rgb,
                                                         const Vector4f& projParams_rgb,
                                                         const ITMFusionParams& fusionParams,
                                                         const ITMSceneParams& sceneParams,
                                                         const float* depth,
                                                         const Vector4f* depthNormals,
                                                         const float* confidence,
                                                         const Vector2i& imgSize_d,
                                                         const Vector4u* rgb,
                                                         const Vector2i& imgSize_rgb)
{
	// Don't carve, when the voxel was updated during fusion of THIS frame
	// (This prevents accidentally carving voxels with rays passing by close to a surface)
	if (voxelRayCastingSum.weightSum > 0)
		return;

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

	// TODO: check which pixels are used
	int x_min = MAX(static_cast<int>(voxelUL_image.x + 0.5f), 1);
	int x_max = MIN(static_cast<int>(voxelBR_image.x + 0.5f), imgSize.x - 2);
	int y_min = MAX(static_cast<int>(voxelUL_image.y + 0.5f), 1);
	int y_max = MIN(static_cast<int>(voxelBR_image.y + 0.5f), imgSize.y - 2);

	/// Check area and carve voxel, if necessary

	// skip positive sdf values (because they either mean free space or are values required to estimate the 0-transition)
	if (ITMVoxel::valueToFloat(voxel.sdf) >= 0 or ITMVoxel::weightToFloat(voxel.w_depth, sceneParams.maxW) <= 0)
		return;

	Matrix4f invM_d;
	M_d.inv(invM_d);
	Vector4f invProjParams_d = invertProjectionParams(projParams_d);

	float carveWeight = 0;
	int count = 0;
	for (int x = x_min; x <= x_max; x++)
		for (int y = y_min; y <= y_max; y++)
		{
			int idx = x + y * imgSize.x;
			// get measured depth from image
			float depthValue = depth[idx];
			Vector4f normal_camera = depthNormals[idx];
			if (depthValue <= 0.0 or normal_camera.w != 1) continue;

			// Normal too steep -> don't carve (because unreliable)
			// discard voxel (return) completely because neighboring normals not reliable
			if (ORUtils::dot(normal_camera.toVector3(), Vector3f(0, 0, -1)) < 0.337)
				return;

			Vector3f normal_world = normalCameraToWorld(normal_camera, invM_d);
			Vector3f rayDirection_camera = reprojectImagePoint(x, y, depthValue, invProjParams_d).normalised();
			Vector3f rayDirection_world = normalCameraToWorld(Vector4f(rayDirection_camera, 0), invM_d);
			float carveSurfaceOffset = fabs(1.0 / dot(normal_world, rayDirection_world)) * sceneParams.mu;

			// check whether voxel needs updating
			float eta = depthValue - pt_camera.z;
			if (eta < carveSurfaceOffset) // Within truncation range -> don't carve
				return;

			carveWeight += 0.01f * weightNormal(normal_camera.toVector3(), rayDirection_camera);
			count++;
		}
	if (carveWeight <= 0)
		return;

	voxelRayCastingSum.update(1, carveWeight / count);
}

_CPU_AND_GPU_CODE_
inline void SetBlockVisibleType(const ITMHashEntry* hashTable,
                                Vector4s* blockCoords,
                                TSDFDirection* blockDirections,
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
		if (hashEntry.ptr >= -1) //search excess list only if there is no room in ordered part
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
inline void SetBlockAllocAndVisibleType(const ITMHashEntry* hashTable,
                                        Vector4s* blockCoords,
                                        TSDFDirection* blockDirections,
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

/**
 * Helper function to access SummingVoxel in SummingVoxelMap given voxelIdx
 * @tparam SummingVoxelMap
 * @param summingVoxelMap
 * @param voxelIdx
 * @return
 */
template<typename SummingVoxelMap>
_CPU_AND_GPU_CODE_
SummingVoxel* getSummingVoxel(SummingVoxelMap& summingVoxelMap, const Vector3s& voxelIdx, TSDFDirection_type directionIdx = 0)
{
	static_assert(std::is_same<typename SummingVoxelMap::key_type, IndexType>::value,
	              "template parameter SummingVoxelMap requires key type IndexType");
	static_assert(std::is_same<typename SummingVoxelMap::mapped_type, SummingVoxel*>::value,
	              "template parameter SummingVoxelMap requires value type SummingVoxel*");

#if __CUDA_ARCH__
	if (summingVoxelMap.bucket_count() <= 0)
		return nullptr;
#endif

	Vector3i blockPos;
	unsigned short offset;
	voxelToBlockPosAndOffset(voxelIdx.toInt(), blockPos, offset);

	auto it = summingVoxelMap.find(IndexType(blockPos.toShort(), directionIdx));
	if (it == summingVoxelMap.end())
		return nullptr;
	else
		return it->second + offset;
}

template<typename SummingVoxelMap>
_CPU_AND_GPU_CODE_
inline void rayCastCarveSpace(int x, int y, Vector2i imgSize, float* depth, Vector4f* depthNormals,
                              const Matrix4f& invM_d,
                              const Vector4f& invProjParams_d, const Vector4f& invProjParams_rgb,
                              const ITMFusionParams& fusionParams,
                              const ITMSceneParams& sceneParams,
                              const ITMHashEntry* hashTable,
                              SummingVoxelMap summingVoxelMap,
                              ITMVoxel* voxelArray
)
{
	static_assert(std::is_same<typename SummingVoxelMap::key_type, IndexType>::value,
	              "template parameter SummingVoxelMap requires key type IndexType");
	static_assert(std::is_same<typename SummingVoxelMap::mapped_type, SummingVoxel*>::value,
	              "template parameter SummingVoxelMap requires value type SummingVoxel*");

	const float mu = sceneParams.mu;
	const float viewFrustum_min = sceneParams.viewFrustum_min;
	const float viewFrustum_max = sceneParams.viewFrustum_max;
	const float voxelSize = sceneParams.voxelSize;

	int idx = y * imgSize.x + x;

	float depthValue = depth[idx];
	Vector4f normal_camera = depthNormals[idx];
	if ((depthValue - mu) < viewFrustum_min or (depthValue + mu) > viewFrustum_max or normal_camera.w != 1)
		return;

	// Normal too steep -> don't carve (because unreliable)
//	if (ORUtils::dot(normal_camera.toVector3(), Vector3f(0, 0, -1)) < 0.337)
//		return;

	Vector3f pt_camera = reprojectImagePoint(x, y, depthValue, invProjParams_d);
	Vector3f pt_world = (invM_d * Vector4f(pt_camera, 1)).toVector3();
	Vector4f rayStart_camera = Vector4f(reprojectImagePoint(x, y, viewFrustum_min, invProjParams_d), 1);
	Vector3f rayStart_world = (invM_d * rayStart_camera).toVector3();
	Vector3f rayDirection_world = (pt_world - rayStart_world).normalised();

	Matrix4f M_d;
	invM_d.inv(M_d);

	Vector3f normal_world = normalCameraToWorld(normal_camera, invM_d);
	float carveDistance = ORUtils::length(pt_world - rayStart_world)
	                      - fabs(1.0 / dot(normal_world, rayDirection_world)) * mu;

	// Decrease distance by depth noise model (std. dev * 3)
	carveDistance -= depthNoiseSigma(depthValue, fabs(1.0 - dot(normal_world, -rayDirection_world)) * M_PI * 0.5) * 3;

	BlockTraversal blockTraversal(rayStart_world, rayDirection_world, carveDistance, voxelSize, true);
	ITMVoxelBlockHash::IndexCache cache[N_DIRECTIONS];
	while (blockTraversal.HasNextBlock())
	{
		Vector3i voxelIdx = blockTraversal.GetNextBlock();

		Vector3i blockPos;
		ushort linearIdx;
		voxelToBlockPosAndOffset(voxelIdx, blockPos, linearIdx);

		Vector3f viewRay_camera = reprojectImagePoint(x, y, 1, invProjParams_d).normalised();
		float weight = 0.01f * weightNormal(normal_camera.toVector3(), viewRay_camera);

		/// find and update voxels
		if (fusionParams.tsdfMode == TSDFMode::TSDFMODE_DIRECTIONAL)
		{
			for (TSDFDirection_type direction = 0; direction < N_DIRECTIONS; direction++)
			{
				int foundEntry = false;
				int index = findVoxel(hashTable, voxelIdx, TSDFDirection(direction), foundEntry, cache[direction]);
				if (not foundEntry)
					continue;

				SummingVoxel* voxel = getSummingVoxel(summingVoxelMap, voxelIdx.toShort(), direction);
				if (not voxel)
					continue;

				// skip positive sdf values (because they either mean free space or are values required to estimate the 0-transition)
				if (voxelArray[index].sdf >= 0 or ITMVoxel::weightToFloat(voxelArray[index].w_depth, sceneParams.maxW) <= 0)
					continue;

				// Don't carve, when the voxel was updated during fusion of THIS frame
				// (This prevents accidentally carving voxels with rays passing by close to a surface)
				if (voxel->weightSum > 0)
					continue;

				voxel->update(1, weight);
			}
		} else
		{
			int foundEntry = false;
			int index = findVoxel(hashTable, voxelIdx, foundEntry, cache[0]);
			if (not foundEntry)
				continue;

			SummingVoxel* voxel = getSummingVoxel(summingVoxelMap, voxelIdx.toShort());
			if (not voxel)
				continue;

			// skip positive sdf values (because they either mean free space or are values required to estimate the 0-transition)
			if (voxelArray[index].sdf >= 0)
				continue;

			// Don't carve, when the voxel was updated during fusion of THIS frame
			// (This prevents accidentally carving voxels with rays passing by close to a surface)
			if (voxel->weightSum > 0)
				continue;

			voxel->update(1, weight);
		}
	}
}

template<typename SummingVoxelMap>
_CPU_AND_GPU_CODE_
inline void rayCastUpdate(int x, int y, Vector2i imgSize_d, Vector2i imgSize_rgb,
                          float* depth, Vector4f* depthNormals, const Vector4u* rgb,
                          const Matrix4f& invM_d, const Matrix4f& M_rgb,
                          const Vector4f& invProjParams_d, const Vector4f& projParams_rgb,
                          const ITMFusionParams& fusionParams,
                          const ITMSceneParams& sceneParams,
                          SummingVoxelMap summingVoxelMap)
{
	static_assert(std::is_same<typename SummingVoxelMap::key_type, IndexType>::value,
	              "template parameter SummingVoxelMap requires key type IndexType");
	static_assert(std::is_same<typename SummingVoxelMap::mapped_type, SummingVoxel*>::value,
	              "template parameter SummingVoxelMap requires value type SummingVoxel*");

	const float mu = sceneParams.mu;
	const float viewFrustum_min = sceneParams.viewFrustum_min;
	const float viewFrustum_max = sceneParams.viewFrustum_max;
	const float voxelSize = sceneParams.voxelSize;

	int idx = y * imgSize_d.x + x;

	float depthValue = depth[idx];
	Vector4f normal_camera = depthNormals[idx];

	if ((depthValue - mu) < viewFrustum_min or (depthValue + mu) > viewFrustum_max or normal_camera.w != 1)
		return;

	// Normal too steep -> don't update (because unreliable)
//	if (ORUtils::dot(normal_camera.toVector3(), Vector3f(0, 0, -1)) < 0.707)
//		return;

	Vector4f pt_camera = Vector4f(reprojectImagePoint(x, y, depthValue, invProjParams_d), 1);
	Vector3f pt_world = (invM_d * pt_camera).toVector3();
	Vector3f normal_world = normalCameraToWorld(normal_camera, invM_d);

	Vector3f pt_rgb_camera = M_rgb * pt_world;
	Vector2f cameraCoordsRGB = project(pt_rgb_camera, projParams_rgb);
	Vector4f color(0, 0, 0, 0);
	if ((cameraCoordsRGB.x >= 1) and (cameraCoordsRGB.x <= imgSize_rgb.x - 2)
	    and (cameraCoordsRGB.y >= 1) and (cameraCoordsRGB.y <= imgSize_rgb.y - 2))
		color = interpolateBilinear(rgb, cameraCoordsRGB, imgSize_rgb);

	Vector3f viewRay_camera = reprojectImagePoint(x, y, 1, invProjParams_d).normalised();

	float angles[N_DIRECTIONS];
	if (fusionParams.tsdfMode == TSDFMode::TSDFMODE_DIRECTIONAL)
	{
		ComputeDirectionAngle(normal_world, angles);
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

	// FIXME: why add block to start?? (its offset without, but why?)
	BlockTraversal blockTraversalBefore(pt_world + Vector3f(1, 1, 1) * voxelSize / 2, rayDirectionBefore, 1.25 * mu,
	                                    voxelSize, true);
	BlockTraversal blockTraversalBehind(pt_world + Vector3f(1, 1, 1) * voxelSize / 2, rayDirectionBehind, mu, voxelSize,
	                                    true);
	if (blockTraversalBehind.HasNextBlock()) blockTraversalBehind.GetNextBlock(); // Skip first voxel to prevent duplicate fusion

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
				if (angles[direction] > direction_angle_threshold)
//				if (angles[direction] > 2 * M_PI_4 or (angles[direction] > direction_angle_threshold and distance <= 0))//direction_angle_threshold) // carve more angle than fusion
				{
					continue;
				}

				SummingVoxel* voxel = getSummingVoxel(summingVoxelMap, voxelIdx.toShort(), direction);
				if (not voxel)
					continue;

				if (fusionParams.useWeighting)
				{
					float directionWeight = DirectionWeight(angles[direction]);
					float voxelDistanceWeight = 1 - MIN(length(voxelSurfaceOffset) / mu, 1);

					weight = combinedWeight(depthValue, distance, normal_camera.toVector3(), viewRay_camera,
					                        sceneParams) * directionWeight;// * voxelDistanceWeight;

					// Normalize by voxel size in camera image (max num rays to hit), so comparable to voxelProjection fusion
					weight *= 1 / (voxelSideLengthCamera * voxelSideLengthCamera);

					// Carving space (parts that exceed direction angle)
					if (angles[direction] > direction_angle_threshold)
						weight = 0.1f * weightNormal(normal_camera.toVector3(), viewRay_camera);
				}

				voxel->update(distance, weight, color.toVector3(), color.w > 0 ? weight : 0);
			}
		} else
		{
			SummingVoxel* voxel = getSummingVoxel(summingVoxelMap, voxelIdx.toShort());
			if (not voxel)
				continue;

			if (fusionParams.useWeighting)
			{
				float voxelDistanceWeight = 1 - MIN(length(voxelSurfaceOffset) / mu, 1);
				weight = combinedWeight(depthValue, distance, normal_camera.toVector3(), viewRay_camera, sceneParams)
				         / powf(voxelSize * 100, 3) * voxelDistanceWeight;

				// Normalize by voxel size in camera image (max num rays to hit), so comparable to voxelProjection fusion
				weight *= 1 / (voxelSideLengthCamera * voxelSideLengthCamera);
			}

			voxel->update(distance, weight, color.toVector3(), color.w > 0 ? weight : 0);
		}
	}
}

/**
 * Collect and combine summed voxels after ray cast update.
 */
_CPU_AND_GPU_CODE_
inline void rayCastCombine(ITMVoxel& voxel, const SummingVoxel& rayCastingSum, const ITMSceneParams& sceneParams)
{
	float deltaSDF = rayCastingSum.sdfSum / rayCastingSum.weightSum;
	float deltaWeight = rayCastingSum.weightSum;
	if (deltaWeight == 0)
		return;

	float currentSDF = ITMVoxel::valueToFloat(voxel.sdf);
	float currentWeight = ITMVoxel::weightToFloat(voxel.w_depth, sceneParams.maxW);

	if (sceneParams.stopIntegratingAtMaxW and currentWeight == sceneParams.maxW)
		return;

	float newWeight = currentWeight + deltaWeight;
	float newSDF = (currentWeight * currentSDF + deltaWeight * deltaSDF) / newWeight;
	newWeight = MIN(newWeight, sceneParams.maxW);

	float currentColorWeight = ITMVoxel::weightToFloat(voxel.w_color, sceneParams.maxW);
	float newColorWeight = currentColorWeight + rayCastingSum.colorWeightSum;
	Vector3u newColor = ((currentColorWeight * voxel.clr.toFloat() + rayCastingSum.colorSum) / newColorWeight).toUChar();

	voxel.sdf = ITMVoxel::floatToValue(newSDF);
	voxel.w_depth = ITMVoxel::floatToWeight(newWeight, sceneParams.maxW);
	voxel.clr = newColor;
	voxel.w_color = ITMVoxel::floatToWeight(MIN(newColorWeight, sceneParams.maxW), sceneParams.maxW);
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
buildSpaceCarvingVisibleType(HashEntryVisibilityType* entriesVisibleType,
                             int x, int y,
                             Vector4s* blockCoords, TSDFDirection* blockDirections,
                             const float* depth,
                             const Vector4f* depthNormal, Matrix4f invM_d,
                             Vector4f projParams_d, float mu, Vector2i imgSize, float voxelSize,
                             const ITMHashEntry* hashTable, float viewFrustum_min, float viewFrustum_max,
                             const ITMFusionParams& fusionParams
)
{
	float depth_measure = depth[x + y * imgSize.x];
	if ((depth_measure - mu) < viewFrustum_min or (depth_measure + mu) > viewFrustum_max)
		return;

	Vector3f pt_camera = reprojectImagePoint(x, y, depth_measure, projParams_d);
	Vector3f pt_world = (invM_d * Vector4f(pt_camera, 1)).toVector3();
	Vector4f rayStart_camera = Vector4f(reprojectImagePoint(x, y, viewFrustum_min, projParams_d), 1);
	Vector3f rayStart_world = (invM_d * rayStart_camera).toVector3();
	Vector3f rayDirection_world = (pt_world - rayStart_world).normalised();

	float carveDistance = ORUtils::length(pt_world - rayStart_world);
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
 * Ray cast depth image to find visible blocks for allocation, insert into set
 *
 * @param visibleBlocks set to insert visible blocks into
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
 * @tparam Set
 */
template<template<typename...> class Set, typename... Args>
_CPU_AND_GPU_CODE_ inline void
findAllocationBlocks(Set<ITMIndexDirectional, Args...>& visibleBlocks,
                     int x, int y, const float* depth, const Vector4f* depthNormal,
                     const Matrix4f& invM_d, Vector4f projParams_d, float mu, Vector2i imgSize, float voxelSize,
                     float viewFrustum_min, float viewFrustum_max, const ITMFusionParams& fusionParams)
{
	float depth_measure = depth[x + y * imgSize.x];
	Vector4f normal_camera = depthNormal[x + y * imgSize.x];
	if ((depth_measure - mu) < viewFrustum_min or (depth_measure + mu) > viewFrustum_max or normal_camera.w != 1)
		return;

	Vector3f normalWorld = normalCameraToWorld(normal_camera, invM_d);

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

	float angles[N_DIRECTIONS];
	ComputeDirectionAngle(normalWorld, angles);

	Vector3s lastBlockIdx(MAX_SHORT, MAX_SHORT, MAX_SHORT);
	while (blockTraversalBefore.HasNextBlock() or blockTraversalBehind.HasNextBlock())
	{
		Vector3i voxelPos;
		if (blockTraversalBefore.HasNextBlock())
			voxelPos = blockTraversalBefore.GetNextBlock();
		else
			voxelPos = blockTraversalBehind.GetNextBlock();

		Vector3s blockIdx = voxelToBlockPos(voxelPos).toShort();
		if (blockIdx == lastBlockIdx)
			continue;

		if (fusionParams.tsdfMode == TSDFMode::TSDFMODE_DIRECTIONAL)
		{
			for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
			{
				if (angles[directionIdx] > M_PI_4)
					continue;
//				if (DirectionWeight(angles[directionIdx]) <= 0)
//					continue;
				visibleBlocks.insert(ITMIndexDirectional(blockIdx, TSDFDirection(directionIdx)));
			}
		} else
		{
			visibleBlocks.insert(ITMIndexDirectional(blockIdx, TSDFDirection::NONE));
		}
		lastBlockIdx = blockIdx;
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
buildHashAllocAndVisibleType(HashEntryAllocType* entriesAllocType,
                             HashEntryVisibilityType* entriesVisibleType,
                             int x, int y,
                             Vector4s* blockCoords, TSDFDirection* blockDirections,
                             const float* depth,
                             const Vector4f* depthNormal, Matrix4f invM_d,
                             Vector4f projParams_d, float mu, Vector2i imgSize, float voxelSize,
                             const ITMHashEntry* hashTable, float viewFrustum_min, float viewFrustum_max,
                             const ITMFusionParams& fusionParams
)
{
	float depth_measure = depth[x + y * imgSize.x];
	Vector4f normal_camera = depthNormal[x + y * imgSize.x];
	if ((depth_measure - mu) < viewFrustum_min or (depth_measure + mu) > viewFrustum_max or normal_camera.w != 1)
		return;

	Vector3f normalWorld = normalCameraToWorld(normal_camera, invM_d);

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

	float angles[N_DIRECTIONS];
	ComputeDirectionAngle(normalWorld, angles);

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
				if (DirectionWeight(angles[direction]) <= 0)
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


template<bool checkEnlarged>
_CPU_AND_GPU_CODE_ inline void checkPointVisibility(bool& isVisible, bool& isVisibleEnlarged,
                                                    const Vector4f& pt_image, const Matrix4f& M_d,
                                                    const Vector4f& projParams_d,
                                                    const Vector2i& imgSize)
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
	} else if (checkEnlarged)
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

template<bool checkEnlarged>
_CPU_AND_GPU_CODE_ inline void checkBlockVisibility(bool& isVisible, bool& isVisibleEnlarged,
                                                    const Vector3s& hashPos, const Matrix4f& M_d,
                                                    const Vector4f& projParams_d,
                                                    const float& voxelSize, const Vector2i& imgSize)
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
	checkPointVisibility<checkEnlarged>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;

	// 0 0 1
	pt_image.z += factor;
	checkPointVisibility<checkEnlarged>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;

	// 0 1 1
	pt_image.y += factor;
	checkPointVisibility<checkEnlarged>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;

	// 1 1 1
	pt_image.x += factor;
	checkPointVisibility<checkEnlarged>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;

	// 1 1 0 
	pt_image.z -= factor;
	checkPointVisibility<checkEnlarged>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;

	// 1 0 0 
	pt_image.y -= factor;
	checkPointVisibility<checkEnlarged>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;

	// 0 1 0
	pt_image.x -= factor;
	pt_image.y += factor;
	checkPointVisibility<checkEnlarged>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;

	// 1 0 1
	pt_image.x += factor;
	pt_image.y -= factor;
	pt_image.z += factor;
	checkPointVisibility<checkEnlarged>(isVisible, isVisibleEnlarged, pt_image, M_d, projParams_d, imgSize);
	if (isVisible) return;
}

} // namespace ITMLib
