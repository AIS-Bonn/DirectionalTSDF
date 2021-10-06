// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <math_constants.h>
#include <ITMLib/Trackers/Shared/ITMICPTracker_Shared.h>
#include <ITMLib/Engines/Reconstruction/Shared/ITMFusionWeight.hpp>
#include "ITMLib/Objects/Scene/ITMDirectional.h"
#include "ITMLib/Objects/Scene/ITMMultiSceneAccess.h"
#include "ITMLib/Objects/Scene/ITMRepresentationAccess.h"
#include "ITMLib/Engines/Reconstruction/Interface/ITMSceneReconstructionEngine.h"
#include "ITMLib/Utils/ITMProjectionUtils.h"
#include "ITMLib/Utils/ITMFilter.h"

namespace ITMLib
{

static const CONSTPTR(int) MAX_RENDERING_BLOCKS = 65536 * 4;
//static const int MAX_RENDERING_BLOCKS = 16384;
static const CONSTPTR(int) minmaximg_subsample = 8;

static const float bilateralFilterSigmaD = 2.0;
static const float bilateralFilterSigmaR = 0.1;
static const int bilateralFilterRadius = 3;

static const float cluster_direction_threshold = M_PI_4;

struct RenderingBlock
{
	Vector2s upperLeft;
	Vector2s lowerRight;
	Vector2f zRange;
};

#ifndef FAR_AWAY
#define FAR_AWAY 999999.9f
#endif

#ifndef VERY_CLOSE
#define VERY_CLOSE 0.05f
#endif

static const CONSTPTR(int) renderingBlockSizeX = 16;
static const CONSTPTR(int) renderingBlockSizeY = 16;

/** Project voxel onto image plane and compute uper left, lower right coordinates as well as zRange
 * @param blockPos
 * @param pose
 * @param intrinsics
 * @param imgSize
 * @param voxelSize
 * @param upperLeft
 * @param lowerRight
 * @param zRange
 * @return true, if within image and z-range
 */
_CPU_AND_GPU_CODE_ inline bool ProjectSingleBlock(const THREADPTR(Vector3s)& blockPos, const THREADPTR(Matrix4f)& pose,
                                                  const THREADPTR(Vector4f)& intrinsics,
                                                  const THREADPTR(Vector2i)& imgSize, float voxelSize,
                                                  THREADPTR(Vector2i)& upperLeft, THREADPTR(Vector2i)& lowerRight,
                                                  THREADPTR(Vector2f)& zRange)
{
	upperLeft = imgSize / minmaximg_subsample;
	lowerRight = Vector2i(-1, -1);
	zRange = Vector2f(FAR_AWAY, VERY_CLOSE);
	for (int corner = 0; corner < 8; ++corner)
	{
		// project all 8 corners down to 2D image
		Vector3s tmp = blockPos;
		tmp.x += (corner & 1) ? 1 : 0;
		tmp.y += (corner & 2) ? 1 : 0;
		tmp.z += (corner & 4) ? 1 : 0;
		Vector4f pt3d(TO_FLOAT3(tmp) * (float) SDF_BLOCK_SIZE * voxelSize, 1.0f);
		pt3d = pose * pt3d;
		if (pt3d.z < 1e-6) continue;

		Vector2f pt2d;
		pt2d.x = (intrinsics.x * pt3d.x / pt3d.z + intrinsics.z) / minmaximg_subsample;
		pt2d.y = (intrinsics.y * pt3d.y / pt3d.z + intrinsics.w) / minmaximg_subsample;

		// remember bounding box, zmin and zmax
		if (upperLeft.x > floor(pt2d.x)) upperLeft.x = (int) floor(pt2d.x);
		if (lowerRight.x < ceil(pt2d.x)) lowerRight.x = (int) ceil(pt2d.x);
		if (upperLeft.y > floor(pt2d.y)) upperLeft.y = (int) floor(pt2d.y);
		if (lowerRight.y < ceil(pt2d.y)) lowerRight.y = (int) ceil(pt2d.y);
		if (zRange.x > pt3d.z) zRange.x = pt3d.z;
		if (zRange.y < pt3d.z) zRange.y = pt3d.z;
	}

	// do some sanity checks and respect image bounds
	if (upperLeft.x < 0) upperLeft.x = 0;
	if (upperLeft.y < 0) upperLeft.y = 0;
	if (lowerRight.x >= imgSize.x) lowerRight.x = imgSize.x - 1;
	if (lowerRight.y >= imgSize.y) lowerRight.y = imgSize.y - 1;
	if (upperLeft.x > lowerRight.x) return false;
	if (upperLeft.y > lowerRight.y) return false;
	//if (zRange.y <= VERY_CLOSE) return false; never seems to happen
	if (zRange.x < VERY_CLOSE) zRange.x = VERY_CLOSE;
	if (zRange.y < VERY_CLOSE) return false;

	return true;
}

_CPU_AND_GPU_CODE_ inline void CreateRenderingBlocks(DEVICEPTR(RenderingBlock)* renderingBlockList, int offset,
                                                     const THREADPTR(Vector2i)& upperLeft,
                                                     const THREADPTR(Vector2i)& lowerRight,
                                                     const THREADPTR(Vector2f)& zRange)
{
	// split bounding box into 16x16 pixel rendering blocks
	for (int by = 0; by < ceil((float) (1 + lowerRight.y - upperLeft.y) / renderingBlockSizeY); ++by)
	{
		for (int bx = 0; bx < ceil((float) (1 + lowerRight.x - upperLeft.x) / renderingBlockSizeX); ++bx)
		{
			if (offset >= MAX_RENDERING_BLOCKS) return;
			//for each rendering block: add it to the list
			DEVICEPTR(RenderingBlock)& b(renderingBlockList[offset++]);
			b.upperLeft.x = upperLeft.x + bx * renderingBlockSizeX;
			b.upperLeft.y = upperLeft.y + by * renderingBlockSizeY;
			b.lowerRight.x = upperLeft.x + (bx + 1) * renderingBlockSizeX - 1;
			b.lowerRight.y = upperLeft.y + (by + 1) * renderingBlockSizeY - 1;
			if (b.lowerRight.x > lowerRight.x) b.lowerRight.x = lowerRight.x;
			if (b.lowerRight.y > lowerRight.y) b.lowerRight.y = lowerRight.y;
			b.zRange = zRange;
		}
	}
}

_CPU_AND_GPU_CODE_ inline int
forwardProjectPixel(Vector4f pixel, const CONSTPTR(Matrix4f)& M, const CONSTPTR(Vector4f)& projParams,
                    const THREADPTR(Vector2i)& imgSize)
{
	pixel.w = 1;
	pixel = M * pixel;

	Vector2f pt_image;
	pt_image.x = projParams.x * pixel.x / pixel.z + projParams.z;
	pt_image.y = projParams.y * pixel.y / pixel.z + projParams.w;

	if ((pt_image.x < 0) || (pt_image.x > imgSize.x - 1) || (pt_image.y < 0) || (pt_image.y > imgSize.y - 1)) return -1;

	return (int) (pt_image.x + 0.5f) + (int) (pt_image.y + 0.5f) * imgSize.x;
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void computeNormalAndAngle(THREADPTR(bool)& foundPoint, const THREADPTR(Vector3f)& point,
                                                     const Vector6f* directionalContribution,
                                                     const CONSTPTR(TVoxel)* voxelBlockData,
                                                     const CONSTPTR(typename TIndex::IndexData)* indexData,
                                                     const THREADPTR(Vector3f)& lightSource,
                                                     THREADPTR(Vector3f)& outNormal, THREADPTR(float)& angle)
{
	if (!foundPoint) return;

	if (directionalContribution)
	{
		outNormal = Vector3f(0, 0, 0);
		for (TSDFDirection_type direction = 0; direction < N_DIRECTIONS; direction++)
		{
			if (directionalContribution->v[direction] > 0)
				outNormal += directionalContribution->v[direction] *
				             computeSingleNormalFromSDF(voxelBlockData, indexData, point,
				                                        TSDFDirection(direction)).normalised();
		}
		outNormal = outNormal.normalised();
	} else
	{
		outNormal = computeSingleNormalFromSDF(voxelBlockData, indexData, point, TSDFDirection::NONE).normalised();
	}

	Vector3f lightDirection = (lightSource - point).normalised();
	angle = dot(outNormal, lightDirection);
	if (!(angle > 0.0)) foundPoint = false;
}

template<bool useSmoothing, bool flipNormals>
_CPU_AND_GPU_CODE_ inline void
computeNormalAndAngle(
	const Vector4f* pointsRay, const Vector4f* normalsRay, const Vector3f& lightSource, const Vector2i& imgSize,
	const int x, const int y,
	bool& foundPoint, Vector3f& outNormal, float& angle)
{
	if (!foundPoint) return;

	Vector4f normal = computeNormalBilateralFiltered(normalsRay, bilateralFilterSigmaD,
	                                                 bilateralFilterSigmaR, bilateralFilterRadius,
	                                                 x, y, imgSize);
	if (normal.w < 0)
	{
		foundPoint = false;
		return;
	}

	outNormal = normal.toVector3();
	if (flipNormals) outNormal = -outNormal;

	Vector3f lightDirection = (lightSource - pointsRay[x + y * imgSize.x].toVector3()).normalised();
	angle = dot(outNormal, lightDirection);
	if (!(angle > 0.0)) foundPoint = false;
}

template<bool useSmoothing, bool flipNormals>
_CPU_AND_GPU_CODE_ inline void
computeNormal(const Vector4f* pointsRay, const float& voxelSize, const Vector2i& imgSize,
							const int& x, const int& y, bool& foundPoint, Vector3f& outNormal)
{
	if (!foundPoint) return;

	int levels = 1;
	if (useSmoothing)
		levels = 2;
	if (y < levels || y > imgSize.y - levels - 1 || x < levels || x > imgSize.x - levels - 1)
	{
		foundPoint = false;
		return;
	}

	// gaussian exp(-i^2 / (2 * sigma^2)) with sigma = 1
	const float weights[6] = {1.0, 0.6065306597126334, 0.1353352832366127, 0.011108996538242306, 0.00033546262790251185,
	                          3.726653172078671e-06};
	Vector3f sum(0, 0, 0);
	float weightSum = 0;

	const Vector4f& x_y = pointsRay[x + y * imgSize.x];

	// define maximum threshold of neighboring points for normal computation
	// use minimum possible distance to points of neighboring pixels as basis (focal length fx=525)
	const float distToCamera = ORUtils::length(x_y.toVector3());
	const float distThreshold = 3 * distToCamera / 525;

	for (int i = 1; i < levels + 1; i++)
	{
		const Vector4f& xp_y = pointsRay[(x + i) + y * imgSize.x];
		const Vector4f& xm_y = pointsRay[(x - i) + y * imgSize.x];
		const Vector4f& x_yp = pointsRay[x + (y + i) * imgSize.x];
		const Vector4f& x_ym = pointsRay[x + (y - i) * imgSize.x];

		if (xp_y.w > 0 and x_yp.w > 0)
		{
			Vector3f diff_x = (xp_y - x_y).toVector3();
			Vector3f diff_y = (x_yp - x_y).toVector3();
			if (ORUtils::length(diff_x) < distThreshold and ORUtils::length(diff_y) < distThreshold)
			{
				weightSum += weights[i - 1];
				Vector3f normal = -ORUtils::cross(diff_x, diff_y).normalised();
				sum += weights[i - 1] * normal;
			}
		}
		if (xm_y.w > 0 and x_yp.w > 0)
		{
			Vector3f diff_x = (x_y - xm_y).toVector3();
			Vector3f diff_y = (x_yp - x_y).toVector3();
			if (ORUtils::length(diff_x) < distThreshold and ORUtils::length(diff_y) < distThreshold)
			{
				weightSum += weights[i - 1];
				Vector3f normal = -ORUtils::cross(diff_x, diff_y).normalised();
				sum += weights[i - 1] * normal;
			}
		}
		if (xm_y.w > 0 and x_ym.w > 0)
		{
			Vector3f diff_x = (x_y - xm_y).toVector3();
			Vector3f diff_y = (x_y - x_ym).toVector3();
			if (ORUtils::length(diff_x) < distThreshold and ORUtils::length(diff_y) < distThreshold)
			{
				weightSum += weights[i - 1];
				Vector3f normal = -ORUtils::cross(diff_x, diff_y).normalised();
				sum += weights[i - 1] * normal;
			}
		}
		if (xp_y.w > 0 and x_ym.w > 0)
		{
			Vector3f diff_x = (xp_y - x_y).toVector3();
			Vector3f diff_y = (x_y - x_ym).toVector3();

			if (ORUtils::length(diff_x) < distThreshold and ORUtils::length(diff_y) < distThreshold)
			{
				weightSum += weights[i - 1];
				Vector3f normal = -ORUtils::cross(diff_x, diff_y).normalised();
				sum += weights[i - 1] * normal;
			}
		}
	}

	if (weightSum < 1)
	{
		foundPoint = false;
		return;
	}

	outNormal = (sum / weightSum).normalised();
	if (flipNormals) outNormal = -outNormal;
}

_CPU_AND_GPU_CODE_ inline void drawPixelGrey(DEVICEPTR(Vector4u)& dest, const THREADPTR(float)& angle)
{
	float outRes = (0.8f * angle + 0.2f) * 255.0f;
	dest = Vector4u((uchar) outRes);
}

_CPU_AND_GPU_CODE_ inline float interpolateCol(float val, float y0, float x0, float y1, float x1)
{
	return (val - x0) * (y1 - y0) / (x1 - x0) + y0;
}

_CPU_AND_GPU_CODE_ inline float baseCol(float val)
{
	if (val <= -0.75f) return 0.0f;
	else if (val <= -0.25f) return interpolateCol(val, 0.0f, -0.75f, 1.0f, -0.25f);
	else if (val <= 0.25f) return 1.0f;
	else if (val <= 0.75f) return interpolateCol(val, 1.0f, 0.25f, 0.0f, 0.75f);
	else return 0.0;
}

/**
 * @param h in [0, 360]
 * @param s in [0, 1]
 * @param v in [0, 1]
 * @return
 */
_CPU_AND_GPU_CODE_ inline
Vector3f HSVtoRGB(const float h, const float s, const float v)
{
	int sector = static_cast<int>(floor(h / 60));
	float f = (h / 60 - sector);
	float p = v * (1 - s);
	float q = v * (1 - s * f);
	float t = v * (1 - s * (1 - f));
	switch(sector)
	{
		case 0:
		case 6:
			return Vector3f(v, t, p);
		case 1:
			return Vector3f(q, v, p);
		case 2:
			return Vector3f(p, v, t);
		case 3:
			return Vector3f(p, q, v);
		case 4:
			return Vector3f(t, p, v);
		case 5:
			return Vector3f(v, p, q);
	}

	return Vector3f(0, 0, 0);
}

/**
 * Paint the pixel wrt. the given normalized confidence (in [0, 1])
 * @param dest
 * @param angle
 * @param normalizedConfidence confidence normalized to [0, 1]
 */
_CPU_AND_GPU_CODE_ inline void
drawPixelConfidence(DEVICEPTR(Vector4u)& dest, const THREADPTR(float)& angle,
                    const THREADPTR(float)& normalizedConfidence)
{
	float confidenceNorm = CLAMP(normalizedConfidence, 0, 1.0f);

	Vector4f color;
	color = Vector4f(HSVtoRGB(confidenceNorm * 120, 1, 1) * 255, 255);

	Vector4f outRes = (0.8f * angle + 0.2f) * color;
	dest = TO_UCHAR4(outRes);
}

_CPU_AND_GPU_CODE_ inline void drawPixelNormal(DEVICEPTR(Vector4u)& dest, const THREADPTR(Vector3f)& normal_obj)
{
	dest.r = (uchar) ((0.3f + (-normal_obj.r + 1.0f) * 0.35f) * 255.0f);
	dest.g = (uchar) ((0.3f + (-normal_obj.g + 1.0f) * 0.35f) * 255.0f);
	dest.b = (uchar) ((0.3f + (-normal_obj.b + 1.0f) * 0.35f) * 255.0f);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void drawPixelColourDirectional(
	DEVICEPTR(Vector4u)& dest, const CONSTPTR(Vector3f)& point, const Vector6f& directionalContribution,
	const float angle, const CONSTPTR(TVoxel)* voxelBlockData, const CONSTPTR(typename TIndex::IndexData)* indexData)
{
	Vector4f color (0, 0, 0, 255);

	// Debugging: color in contributing directions
	for (TSDFDirection_type direction = 0; direction < N_DIRECTIONS; direction++)
	{
		const Vector3f& clr = TSDFDirectionColor[direction];
		color.x += MIN(clr.x * 255.0f * directionalContribution[direction], 255.0f);
		color.y += MIN(clr.y * 255.0f * directionalContribution[direction], 255.0f);
		color.z += MIN(clr.z * 255.0f * directionalContribution[direction], 255.0f);
	}

	// Debugging: color in strongest contributing direction
//	int maxIdx = -1;
//	float maxWeight = 0;
//	for (TSDFDirection_type direction = 0; direction < N_DIRECTIONS; direction++)
//	{
//		if (directionalContribution[direction] > maxWeight)
//		{
//			maxIdx = direction;
//			maxWeight = directionalContribution[direction];
//		}
//	}
//	const Vector3f &clr = directionColors[maxIdx];
//	dest.x = (uchar) (clr.x * 255.0f);
//	dest.y = (uchar) (clr.y * 255.0f);
//	dest.z = (uchar) (clr.z * 255.0f);

//	for (TSDFDirection_type direction = 0; direction < N_DIRECTIONS; direction++)
//	{
//		Vector4f clr = VoxelColorReader<TVoxel::hasColorInformation, TVoxel, TIndex>::interpolate(
//			voxelBlockData, indexData, point, TSDFDirection(direction));
//		dest.x += (uchar) (clr.x * 255.0f * directionalContribution[direction]);
//		dest.y += (uchar) (clr.y * 255.0f * directionalContribution[direction]);
//		dest.z += (uchar) (clr.z * 255.0f * directionalContribution[direction]);
//	}

	Vector4f outRes = (0.8f * angle + 0.2f) * color;
	dest = TO_UCHAR4(outRes);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void drawPixelColourDefault(DEVICEPTR(Vector4u)& dest, const CONSTPTR(Vector3f)& point,
                                                      const float angle, const CONSTPTR(TVoxel)* voxelBlockData,
                                                      const CONSTPTR(typename TIndex::IndexData)* indexData)
{
	Vector4f clr = VoxelColorReader<TVoxel::hasColorInformation, TVoxel, TIndex>::interpolate(voxelBlockData, indexData,
	                                                                                          point, TSDFDirection::NONE);

	dest.x = (uchar) (clr.x * 255.0f);
	dest.y = (uchar) (clr.y * 255.0f);
	dest.z = (uchar) (clr.z * 255.0f);
	dest.w = 255;
}

template<typename TIndex, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline bool castRayDefaultTSDF(Vector4f& pt_out,
                                                  float& distance_out,
                                                  int x, int y, const Map<TIndex, ITMVoxel*, Args...>& tsdf,
                                                  Matrix4f invM, Vector4f invProjParams,
                                                  const ITMSceneParams& sceneParams,
                                                  const CONSTPTR(Vector2f)& minMaxImg,
                                                  const TSDFDirection direction = TSDFDirection::NONE,
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
		sdfValue = readFromSDF_float_uninterpolated(found, tsdf, pt_result, direction);

		ITMVoxel res = readVoxel(found, tsdf, Vector3i((int) ROUND(pt_result.x), (int) ROUND(pt_result.y), (int) ROUND(pt_result.z)),
		                         direction);
		TIndex index;
		unsigned short linearIdx;
		voxelIdxToIndexAndOffset(index, linearIdx, Vector3i((int) ROUND(pt_result.x), (int) ROUND(pt_result.y), (int) ROUND(pt_result.z)), direction);

		if (!found)
		{
			stepLength = SDF_BLOCK_SIZE;
		} else
		{
			if ((sdfValue <= 0.1f) && (sdfValue >= -0.5f))
			{
				sdfValue = readFromSDF_float_interpolated(found, tsdf, pt_result, direction);
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

		sdfValue = readWithConfidenceFromSDF_float_interpolated(found, confidence, tsdf, pt_result, sceneParams.maxW, direction);

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

	distance_out = totalLength * sceneParams.voxelSize;
	pt_out = Vector4f(pt_result * sceneParams.voxelSize, confidence);

	return true;
}
//
template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline bool castRayDefault(DEVICEPTR(Vector4f)& pt_out,
                                              float& distance_out,
                                              DEVICEPTR(HashEntryVisibilityType)* entriesVisibleType,
                                              int x, int y, const CONSTPTR(TVoxel)* voxelData,
                                              const CONSTPTR(typename TIndex::IndexData)* voxelIndex,
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
		sdfValue = readFromSDF_float_uninterpolated(voxelData, voxelIndex, pt_result, direction, vmIndex, cache);

		if (entriesVisibleType)
		{
			if (vmIndex) entriesVisibleType[vmIndex - 1] = VISIBLE_IN_MEMORY;
		}

		if (!vmIndex)
		{
			stepLength = SDF_BLOCK_SIZE;
		} else
		{
			if ((sdfValue <= 0.1f) && (sdfValue >= -0.5f))
			{
				sdfValue = readFromSDF_float_interpolated(voxelData, voxelIndex, pt_result, direction, vmIndex, cache);
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

		sdfValue = readWithConfidenceFromSDF_float_interpolated(confidence, voxelData, voxelIndex,
		                                                        pt_result, direction, sceneParams.maxW,
		                                                        vmIndex, cache);

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

	distance_out = totalLength * sceneParams.voxelSize;
	// multiply by transition: negative transition <=> negative confidence!
	pt_out = Vector4f(pt_result * sceneParams.voxelSize, confidence);

	return true;
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline bool castRayDirectional2(DEVICEPTR(Vector4f)& pt_out, Vector6f* directionalContribution,
                                                  DEVICEPTR(HashEntryVisibilityType)* entriesVisibleType,
                                                  int x, int y, const CONSTPTR(TVoxel)* voxelData,
                                                  const CONSTPTR(typename TIndex::IndexData)* voxelIndex,
                                                  Matrix4f invM, Vector4f invProjParams,
                                                  const ITMSceneParams& sceneParams,
                                                  const CONSTPTR(Vector2f)& minMaxImg)
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

	pt_camera_f = Vector4f(reprojectImagePoint(x, y, minMaxImg.y,
	                                           invProjParams), 1.0f);
	totalLengthMax = length(TO_VECTOR3(pt_camera_f)) * sceneParams.oneOverVoxelSize;
	pt_block_e = TO_VECTOR3(invM * pt_camera_f) * sceneParams.oneOverVoxelSize;

	rayDirection = (pt_block_e - pt_block_s).normalised();

	pt_result = pt_block_s;

	typename TIndex::IndexCache cache[N_DIRECTIONS];


	float lastSDFValue = sdfValue;
	while (totalLength < totalLengthMax)
	{
		float weightSum = 0;
		sdfValue = 0;
		vmIndex = false;
		for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
		{
			const auto direction = TSDFDirection(directionIdx);
			float confidence_ = 0;
			float sdfValue_ = 1;
			int vmIndex_ = 0;
			sdfValue_ = readWithConfidenceFromSDF_float_interpolated(confidence, voxelData, voxelIndex,
			                                                         pt_result, TSDFDirection(directionIdx),
			                                                         sceneParams.maxW,
			                                                         vmIndex, cache[directionIdx]);
			Vector3f gradient_ = computeSingleNormalFromSDF(voxelData, voxelIndex, pt_result, direction);

			float weight = DirectionWeight(DirectionAngle(gradient_, direction))
			               * ORUtils::dot(gradient_, -rayDirection);// * confidence_;

			weight = MAX(weight, 0);

			sdfValue += weight * sdfValue_;
			weightSum += weight;

		}
		if(weightSum > 0)
		{
			vmIndex = true;
			sdfValue /= weightSum;
			confidence = weightSum;
		}
		else
		{
			sdfValue = 1;
			confidence = 0;
		}

		if (!vmIndex)
		{
			stepLength = SDF_BLOCK_SIZE;
		} else
		{
			if (lastSDFValue > 0 and sdfValue <= 0)
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

		float weightSum = 0;
		sdfValue = 0;
		bool isFreeSpace = false;
		vmIndex = false;
		for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
		{
			const auto direction = TSDFDirection(directionIdx);
			float confidence_ = 0;
			float sdfValue_ = 1;
			int vmIndex_ = 0;
			sdfValue_ = readWithConfidenceFromSDF_float_interpolated(confidence, voxelData, voxelIndex,
			                                                         pt_result, TSDFDirection(directionIdx),
			                                                         sceneParams.maxW,
			                                                         vmIndex, cache[directionIdx]);
			Vector3f gradient_ = computeSingleNormalFromSDF(voxelData, voxelIndex, pt_result, direction);

			float weight = DirectionWeight(DirectionAngle(gradient_, direction))
			 * ORUtils::dot(gradient_, -rayDirection);// * confidence_;

			if (confidence_ > 0 and sdfValue_ > 0.25)
				isFreeSpace = true;

			weight = MAX(weight, 0);

			directionalContribution->v[directionIdx] = weight;

			sdfValue += weight * sdfValue_;
			weightSum += weight;

		}
		if(weightSum > 0 and not isFreeSpace)
		{
			vmIndex = true;
			sdfValue /= weightSum;
			confidence = weightSum;

			if (directionalContribution)
			{
				for (int i = 0; i < 6; i++)
				{
					directionalContribution->v[i] = MAX(directionalContribution->v[i] / weightSum, 0);
				}
			}
		}
		else
		{
			sdfValue = 1;
			confidence = 0;
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
		return false;
	}

	// multiply by transition: negative transition <=> negative confidence!
	pt_out = Vector4f(pt_result * sceneParams.voxelSize, confidence);

	return true;
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline bool castRayDirectional(DEVICEPTR(Vector4f)& pt_out, Vector6f* directionalContribution,
                                                   DEVICEPTR(HashEntryVisibilityType)* entriesVisibleType,
                                                   int x, int y, const CONSTPTR(TVoxel)* voxelData,
                                                   const CONSTPTR(typename TIndex::IndexData)* voxelIndex,
                                                   Matrix4f invM, Vector4f invProjParams,
                                                   const ITMSceneParams& sceneParams,
                                                   const CONSTPTR(Vector2f)& minMaxImg)
{
	Vector4f pt_camera_f;
	Vector3f pt_block_s, pt_block_e, rayDirection, pt_result;
	float totalLength, totalLengthMax, stepScale;

	pt_out = Vector4f(0, 0, 0, 0);

	stepScale = sceneParams.mu * sceneParams.oneOverVoxelSize;

	pt_camera_f = Vector4f(reprojectImagePoint(x, y, minMaxImg.x, invProjParams), 1.0f);
	totalLength = length(TO_VECTOR3(pt_camera_f)) * sceneParams.oneOverVoxelSize;
	pt_block_s = TO_VECTOR3(invM * pt_camera_f) * sceneParams.oneOverVoxelSize;

	pt_camera_f = Vector4f(reprojectImagePoint(x, y,
	                                           minMaxImg.y,
	                                           invProjParams), 1.0f);
	totalLengthMax = length(TO_VECTOR3(pt_camera_f)) * sceneParams.oneOverVoxelSize;
	pt_block_e = TO_VECTOR3(invM * pt_camera_f) * sceneParams.oneOverVoxelSize;

	rayDirection = (pt_block_e - pt_block_s).normalised();
	pt_result = pt_block_s;

	float maxDistancePerVoxel = sceneParams.voxelSize / sceneParams.mu;

	// 1) Find first zero crossing with positive confidence (or use first crossing if all confidences = 0)
	int vmIndex;
	float sdfValues[N_DIRECTIONS] = {1, 1, 1, 1, 1, 1};
	float lastSDFValues[N_DIRECTIONS] = {1, 1, 1, 1, 1, 1};
	typename TIndex::IndexCache cache[N_DIRECTIONS];
	float stepLength = SDF_BLOCK_SIZE;
	auto firstCrossingDirection = static_cast<TSDFDirection_type>(TSDFDirection::NONE);
	float firstCrossingLength = 0;
	Vector3f pt_firstCrossing;
	while (totalLength < totalLengthMax)
	{
		bool foundZeroCrossing = false;
		bool isFreeSpace = false;
		int numReadableValues = 0;
		for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
		{
			const auto direction = TSDFDirection(directionIdx);

			float confidence = 0;
			sdfValues[directionIdx] = readWithConfidenceFromSDF_float_uninterpolated(confidence, voxelData, voxelIndex,
			                                                                         pt_result, direction, sceneParams.maxW,
			                                                                         vmIndex, cache[directionIdx]);
			if (confidence > 0 and sdfValues[directionIdx] > maxDistancePerVoxel) isFreeSpace = true;

			if (vmIndex)
			{
				numReadableValues++;
				if (sdfValues[directionIdx] <= 0.5 and sdfValues[directionIdx] >= -0.5)
				{
					sdfValues[directionIdx] = readWithConfidenceFromSDF_float_interpolated(confidence, voxelData, voxelIndex,
					                                                                       pt_result, TSDFDirection(directionIdx),
					                                                                       sceneParams.maxW,
					                                                                       vmIndex, cache[directionIdx]);
					if (confidence > 0 and sdfValues[directionIdx] > maxDistancePerVoxel) isFreeSpace = true;
				}
				if (lastSDFValues[directionIdx] > 0 and sdfValues[directionIdx] <= 0)
				{
//					Vector3f ptGradient = pt_result + rayDirection * sdfValues[directionIdx] * stepScale; // closer to surface, so gradient computation always works
//					Vector3f gradient = computeSingleNormalFromSDF(voxelData, voxelIndex,
//					                                               ptGradient, TSDFDirection(directionIdx)).normalised();
//					// Check if surface towards camera and compatible with direction
//					bool isApplicable = SIGN(dot(gradient, -rayDirection)) > 0 and
//					                    DirectionAngle(gradient, TSDFDirection(directionIdx)) < direction_angle_threshold;
					bool isApplicable = true;

					// Select first crossing with confidence > 0 or just first crossing if all confidences == 0
					if (firstCrossingDirection >= N_DIRECTIONS and isApplicable)
					{
						pt_firstCrossing = pt_result;
						firstCrossingDirection = directionIdx;
						firstCrossingLength = totalLength;
					}
					if (confidence > 0 and isApplicable)
					{
						pt_firstCrossing = pt_result;
						firstCrossingDirection = directionIdx;
						firstCrossingLength = totalLength;
						foundZeroCrossing = true;
//						break; // don't break to allow free space detection
					}
				}
				stepLength = MIN(stepLength, MAX(sdfValues[directionIdx] * stepScale, 1.0f));
			}
			lastSDFValues[directionIdx] = sdfValues[directionIdx];
		}

		if (foundZeroCrossing and not isFreeSpace)
			break;

		if (isFreeSpace)
		{
			firstCrossingDirection = static_cast<TSDFDirection_type>(TSDFDirection::NONE);
		}

		pt_result += stepLength * rayDirection;
		totalLength += stepLength;

		if (numReadableValues == 0)
			stepLength = SDF_BLOCK_SIZE;
	}

	totalLength = firstCrossingLength;
	totalLengthMax = MIN(totalLengthMax, totalLength + SDF_BLOCK_SIZE);

	pt_result = pt_firstCrossing;

	if (firstCrossingDirection >= N_DIRECTIONS)
		return false;

	firstCrossingDirection = static_cast<TSDFDirection_type>(TSDFDirection::Z_NEG);

	// 2) Determine best direction (via confidence & gradient)
	float weights[N_DIRECTIONS] = {1, 1, 1, 1, 1, 1};
//	float weights[N_DIRECTIONS] = {0, 0, 0, 0, 0, 0};
	Vector3f gradients[N_DIRECTIONS];
	float maxConfidence = 0;
	auto maxConfidenceIdx = firstCrossingDirection;
	for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
	{
		float confidence = 0;
		sdfValues[directionIdx] = readWithConfidenceFromSDF_float_interpolated(confidence, voxelData, voxelIndex, pt_result,
		                                                                       TSDFDirection(directionIdx),
		                                                                       sceneParams.maxW,
		                                                                       vmIndex, cache[directionIdx]);
		if (confidence == 0)// and directionIdx != firstCrossingDirection)
		{
			weights[directionIdx] = 0;
			continue;
		}
		Vector3f ptGradient = pt_result + rayDirection * sdfValues[directionIdx] * stepScale; // closer to surface, so gradient computation always works
		gradients[directionIdx] = computeSingleNormalFromSDF(voxelData, voxelIndex, ptGradient, TSDFDirection(directionIdx));
		// Compliance of surface gradient wrt direction
//		confidence *= DirectionAngle(gradients[directionIdx], TSDFDirection(directionIdx));
//		confidence *= DirectionWeight(DirectionAngle(gradients[directionIdx], TSDFDirection(directionIdx)));
		// Exclude points on opposite side
		confidence *= SIGN(dot(gradients[directionIdx], -rayDirection));

		if (confidence > maxConfidence)
		{
			maxConfidence = confidence;
			maxConfidenceIdx = directionIdx;
		} else if (confidence <= 0)
		{
			weights[directionIdx] = 0;
		}
	}

	if (maxConfidenceIdx >= N_DIRECTIONS)
		return false;

	float sdfValue = sdfValues[maxConfidenceIdx];

	// 3) Determine directions and weights to use for final interpolation (surface gradient)
	float angles[N_DIRECTIONS];
	ComputeDirectionAngle(gradients[maxConfidenceIdx], angles);

	for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
	{
		if (weights[directionIdx] <= 0)
			continue;

//		weights[directionIdx] = DirectionWeight(angles[directionIdx]) * dot(gradients[directionIdx], gradients[maxConfidenceIdx]);
		weights[directionIdx] = dot(gradients[directionIdx], gradients[maxConfidenceIdx]);
		if (weights[directionIdx] != weights[directionIdx])
			weights[directionIdx] = 0; // check NaN
	}

	// Adapt step scale to angle between surface and observation ray
	stepScale = sceneParams.mu * sceneParams.oneOverVoxelSize / MAX(fabs(dot(gradients[maxConfidenceIdx], rayDirection)), 0.5);

	pt_result += rayDirection * sdfValues[maxConfidenceIdx] * stepScale; // closer to surface, so gradient computation always works

	// 4) Perform additional steps to get more accurate zero crossing (interpolate directions)
	float contributionWeights[N_DIRECTIONS] = {0, 0, 0, 0, 0, 0};
	float lastSDFValue;
	float weightSum = 0;
	for (int i = 0; i < 2 or (i < 5 and fabs(sdfValue) > 1e-6); i++)
	{
		lastSDFValue = sdfValue;
		sdfValue = 0;
		weightSum = 0;
		for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
		{
			if (weights[directionIdx] <= 0) continue;

			float confidence = 0;
			float value = readWithConfidenceFromSDF_float_interpolated(confidence, voxelData, voxelIndex, pt_result,
			                                                           TSDFDirection(directionIdx),
			                                                           sceneParams.maxW,
			                                                           vmIndex, cache[directionIdx]);

			if (confidence == 0 and directionIdx != maxConfidenceIdx)
			{
				weights[directionIdx] = 0; // Direction not applicable for estimation
				contributionWeights[directionIdx] = 0;
				continue;
			} else
			{
				confidence = MAX(confidence, 1e-6);
			}

			contributionWeights[directionIdx] = confidence * weights[directionIdx];

			sdfValue += contributionWeights[directionIdx] * value;
			weightSum += contributionWeights[directionIdx];
		}
		if (weightSum <= 0)
			return false;

		sdfValue /= weightSum;
		float reductionFactor = (fabs(sdfValue) - fabs(lastSDFValue)) / fabs(lastSDFValue);
		// rF < 0: magnitude reduction, rF > 0: magnitude increase
		if (SIGN(sdfValue) != SIGN(lastSDFValue) and reductionFactor > -0.75)
		{
			stepScale *= 0.5;
		}

		stepLength = sdfValue * stepScale;
		pt_result += stepLength * rayDirection;
		totalLength += stepLength;

		// Prevent searching too far behind first zero crossing
		if (totalLength > totalLengthMax)
			break;
	}
	if (fabs(sdfValue) > 0.1)// or totalLength > totalLengthMax) // no convergence
	{
		return false;
	}

	// 5) Check all directions we'd suppose to see the surface. If it is not there -> don't render!
	for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
	{
//		if (angles[directionIdx] < cluster_direction_threshold) continue;

		// Only check directions for conflict which haven't been used for estimate
		if (contributionWeights[directionIdx] > 0) continue;

		float confidence = 0;
		float value = readWithConfidenceFromSDF_float_interpolated(confidence, voxelData, voxelIndex, pt_result,
		                                                           TSDFDirection(directionIdx),
		                                                           sceneParams.maxW,
		                                                           vmIndex, cache[directionIdx]);
		if (confidence <= 0) continue;

		float sdfThreshold = 0.5f / (sceneParams.mu / sceneParams.voxelSize); // allow maximum half a voxel size of distance to the surface
		if (value > sdfThreshold)
		{
			return false;
		}

		// Directions which could concur with the surface normal taken more seriously
//		if (angles[directionIdx] < cluster_direction_threshold and value > 0)
//		{
//			return false;
//		}
	}

	if (directionalContribution)
	{
		for (int i = 0; i < 6; i++)
		{
			directionalContribution->v[i] = MAX(contributionWeights[i] / weightSum, 0);
		}
	}

	pt_out = Vector4f(pt_result * sceneParams.voxelSize, weightSum);
	return true;
}

template<class TIndex, class TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline bool castRay(DEVICEPTR(Vector4f)& pt_out, Vector6f* directionalContribution,
                                       int x, int y, const Map<TIndex, TVoxel*, Args...>& tsdf,
                                       Matrix4f invM, Vector4f invProjParams,
                                       const ITMSceneParams& sceneParams,
                                       const Vector2f& minMaxImg,
                                       bool directionalTSDF)
{
	float distance;
	return castRayDefaultTSDF(pt_out, distance, x, y, tsdf, invM,
																				invProjParams, sceneParams, minMaxImg);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void processPixelICP(DEVICEPTR(Vector4f)& pointsMap, DEVICEPTR(Vector4f)& normalsMap,
                                               const THREADPTR(Vector3f)& point, bool foundPoint,
                                               const CONSTPTR(TVoxel)* voxelData,
                                               const CONSTPTR(typename TIndex::IndexData)* voxelIndex,
                                               float voxelSize, const THREADPTR(Vector3f)& lightSource)
{
	Vector3f outNormal;
	float angle;

	computeNormalAndAngle<TVoxel, TIndex>(foundPoint, point, voxelData, voxelIndex, lightSource, outNormal, angle);

	if (foundPoint)
	{
		Vector4f outPoint4;
		outPoint4.x = point.x;
		outPoint4.y = point.y;
		outPoint4.z = point.z;
		outPoint4.w = 1.0f;
		pointsMap = outPoint4;

		Vector4f outNormal4;
		outNormal4.x = outNormal.x;
		outNormal4.y = outNormal.y;
		outNormal4.z = outNormal.z;
		outNormal4.w = 0.0f;
		normalsMap = outNormal4;
	} else
	{
		Vector4f out4;
		out4.x = 0.0f;
		out4.y = 0.0f;
		out4.z = 0.0f;
		out4.w = -1.0f;

		pointsMap = out4;
		normalsMap = out4;
	}
}

template<typename T>
_CPU_AND_GPU_CODE_
inline void insertionSort(const T* in, T* out, int* outIds, size_t length)
{
	for (size_t idx = 0; idx < length; idx++)
	{
		T pivot = in[idx];
		int j = idx - 1;
		for (; j >= 0 and (out[j] > pivot or out[j] < 0); j--)
		{
			out[j + 1] = out[j];
			outIds[j + 1] = outIds[j];
		}
		out[j + 1] = pivot;
		outIds[j + 1] = idx;
	}

}

_CPU_AND_GPU_CODE_
inline void findBestCluster(const float* distances, const float* confidences, const Vector3f* normals,
                            int* clusterOutput, Vector3f& clusterNormalOutput)
{
	float distancesSorted[N_DIRECTIONS];
	int distancesSortedIds[N_DIRECTIONS] = {-1, -1, -1, -1, -1, -1};
	insertionSort(distances, distancesSorted, distancesSortedIds, N_DIRECTIONS);

	/** number indicates starting index of a cluster (end indicated by next cluster or -1)
	 * index corresponds to distancesSorted */
	int clusters[N_DIRECTIONS] = {-1, -1, -1, -1, -1, -1};

	/** confidences for each cluster */
	float clusterConfidences[N_DIRECTIONS] = {0, 0, 0, 0, 0, 0};

	/** representative normal for each cluster */
	Vector3f clusterNormals[N_DIRECTIONS];

	uchar numClusters = 0;
	float lastDistance = -1;
	float interClusterMaxConfidence = -1;
	float interClusterCenterDistanceSum = 0;
	float sumConfidences = 0;

	/// 1. cluster the distances
	for (uchar idx = 0; idx < N_DIRECTIONS and distancesSorted[idx] >= 0 and distancesSorted[idx] < INF_FLOAT; idx++)
	{
		float interClusterCenterDistance = lastDistance;
		if (clusterConfidences[numClusters] > 0)
			interClusterCenterDistance = interClusterCenterDistanceSum / clusterConfidences[numClusters];

		const double maxClusterRadius = 2;
		if (fabs(distancesSorted[idx] - interClusterCenterDistance) > maxClusterRadius
				or distancesSorted[idx] - distancesSorted[clusters[numClusters - 1]] > 2 * maxClusterRadius
		    or ORUtils::dot(clusterNormals[numClusters - 1], normals[distancesSortedIds[idx]]) < 0)//cluster_direction_threshold) // maximum deviation of normal
		{
			numClusters++;
			clusters[numClusters - 1] = idx;
			clusterNormals[numClusters - 1] = normals[distancesSortedIds[idx]];
			interClusterMaxConfidence = -1;
			interClusterCenterDistanceSum = 0;
		}
		lastDistance = distancesSorted[idx];

		float confidence = confidences[distancesSortedIds[idx]];
		clusterConfidences[numClusters - 1] += confidence;
		interClusterCenterDistanceSum += confidence * distancesSorted[idx];

		if (confidence > interClusterMaxConfidence)
		{
			clusterNormals[numClusters - 1] = normals[distancesSortedIds[idx]];
			interClusterMaxConfidence = confidence;
		}

		sumConfidences += confidence;
	}

	int bestCluster = -1;
	for (uchar idx = 0; idx < N_DIRECTIONS and clusters[idx] >= 0; idx++)
	{
		bool skip = false;
		/// Check if there is a more distant cluster containing point from a direction which complies with this cluster's normal
		/// -> this is likely to be a false positive (successfully reduces overhanging edges)
		if (idx + 1 < N_DIRECTIONS and clusters[idx + 1] >= 0)
		{
			for (uchar j = clusters[idx + 1]; j < N_DIRECTIONS and distancesSorted[j] > 0 and distancesSorted[j] < INF_FLOAT; j++)
			{
				if (//confidences[distancesSortedIds[j]] > 0 and
					DirectionAngle(clusterNormals[idx], TSDFDirection(distancesSortedIds[j])) > cluster_direction_threshold)
				{
					skip = true;
					break;
				}
			}
		}

		// First cluster with positive confidence
		if (not skip and clusterConfidences[idx] > 0)
		{
			bestCluster = idx;
			break;
		}
	}

	// If no cluster, select first one
	if (bestCluster < 0 and clusters[0] >= 0)
	{
		bestCluster = 0;
	}

	for (uchar i = 0; i < N_DIRECTIONS; i++)
		clusterOutput[i] = -1;

	if (bestCluster < 0 or clusterConfidences[bestCluster] <= 0)
	{
		clusterNormalOutput = Vector3f(0, 0, 0);
		return;
	}
	char startIdx = clusters[bestCluster];
	char endIdx = clusters[bestCluster + 1] > 0 ? clusters[bestCluster + 1] : N_DIRECTIONS;
	for (uchar idx = startIdx; idx < endIdx and distancesSorted[idx] >= 0; idx++)
	{
		clusterOutput[idx - startIdx] = distancesSortedIds[idx];
	}
	clusterNormalOutput = clusterNormals[bestCluster];
}

struct InputPointClouds
{
	Vector4f* pointCloud[N_DIRECTIONS];
	Vector4f* pointCloudNormals[N_DIRECTIONS];
};

template<bool useSmoothing, bool flipNormals>
_CPU_AND_GPU_CODE_ inline void combineDirectionalPointClouds(
	DEVICEPTR(Vector4f)* outputPointCloud,
	DEVICEPTR(Vector4f)* outputNormals,
	const InputPointClouds& inputPointClouds,
	DEVICEPTR(Vector6f)* directionalContribution,
	const THREADPTR(Vector2i)& imgSize,
	const Matrix4f& invM, const Vector4f& invProjParams,
	const THREADPTR(int)& x, const THREADPTR(int)& y, float voxelSize)
{
	int locId = x + y * imgSize.x;

	outputPointCloud[locId] = Vector4f(0, 0, 0, 0);
	outputNormals[locId] = Vector4f(0, 0, 0, 0);

	Vector4f points[N_DIRECTIONS];
	Vector3f normals[N_DIRECTIONS];
	float confidences[N_DIRECTIONS] = {0, 0, 0, 0, 0, 0};

	float distances[N_DIRECTIONS] = {INF_FLOAT, INF_FLOAT, INF_FLOAT, INF_FLOAT, INF_FLOAT, INF_FLOAT};

	Vector3f viewRay = (invM * Vector4f(reprojectImagePoint(x, y, 1, invProjParams), 0)).toVector3().normalised();
	Vector3f cameraPos_world = (invM * Vector4f(0, 0, 0, 1)).toVector3() / voxelSize; // camera pos in voxel coordinates

	/// Determine normals and confidences, find best candidate
	float sumConfidences = 0;
	for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
	{
		Vector4f& point = points[directionIdx];
		Vector3f& normal = normals[directionIdx];
		float& confidence = confidences[directionIdx];
		float& distance = distances[directionIdx];

		directionalContribution[locId].v[directionIdx] = 0;

		point = inputPointClouds.pointCloud[directionIdx][locId];
		confidence = point.w ;

		bool foundPoint = confidence > 0.0f;
		if (not foundPoint)
			continue;

		distance = length(cameraPos_world - point.toVector3());

		float angle; // becomes dot(-viewRay, normal)
		computeNormalAndAngle<useSmoothing, flipNormals>(inputPointClouds.pointCloud[directionIdx],
		                                                 inputPointClouds.pointCloudNormals[directionIdx],
		                                                 cameraPos_world, imgSize, x, y,
		                                                 foundPoint, normal, angle);

		if (not foundPoint or DirectionAngle(normal, TSDFDirection(directionIdx)) > direction_angle_threshold)
		{
			confidence = 0;
			continue;
		}

		// Compliance of surface gradient wrt direction
//		confidence *= DirectionWeight(angle); // DirectionAngle(normal, TSDFDirection(directionIdx));
		// Exclude points on opposite side
		confidence *= SIGN(dot(normal, -viewRay));

		sumConfidences += confidence > 0 ? confidence : 0;
	}

	if (sumConfidences <= 0)
		return;

	int cluster[N_DIRECTIONS];
	Vector3f clusterNormal;
	findBestCluster(distances, confidences, normals, cluster, clusterNormal);

	if (cluster[0] < 0)
		return;
//
//	 /// Combine cluster to surface point
//	 bool clusterHasGoodDirection = false;
//	 int bestDirection = -1;
//	 float bestDirectionConfidence = 0;
//	 for (uchar idx = 0; idx < N_DIRECTIONS and cluster[idx] >= 0; idx++)
//	 {
//					 TSDFDirection_type directionIdx = cluster[idx];
////           if (DirectionWeight(-viewRay, TSDFDirection(directionIdx)) >= direction_weight_threshold)
//					 if (DirectionAngle(clusterNormal, TSDFDirection(directionIdx)) >= direction_angle_threshold)
//					 {
//									 clusterHasGoodDirection = true;
//									 float confidence = confidences[directionIdx] * DirectionAngle(normals[directionIdx], TSDFDirection(directionIdx));
//									 if (confidences[directionIdx] > 0 and confidence > bestDirectionConfidence)
//									 {
//													 bestDirectionConfidence = confidence;
//													 bestDirection = directionIdx;
//									 }
//					 }
//	 }
//	 if (bestDirection >= 0)
//	 {
//					 outputPointCloud[locId] = points[bestDirection];
//					 outputNormals[locId] = Vector4f(normals[bestDirection], 1);
//					 directionalContribution[locId][bestDirection] = 1.0;
//					 return;
//	 }

	float sumWeights = 0;
	Vector4f sumPoints(0, 0, 0, 0);
	Vector3f sumNormals(0, 0, 0);
	for (uchar idx = 0; idx < N_DIRECTIONS and cluster[idx] >= 0; idx++)
	{
		TSDFDirection_type directionIdx = cluster[idx];

		Vector4f& point = points[directionIdx];
		Vector3f& normal = normals[directionIdx];
		float& confidence = confidences[directionIdx];

		if (confidence <= 0)
			continue;

		// Filter directions, that don't apply to gradient
		if (DirectionAngle(clusterNormal, TSDFDirection(directionIdx)) > direction_angle_threshold)
			continue;

		// Filter gradients, that are too different
		if (dot(normal, clusterNormal) < cluster_direction_threshold)
			continue;

		float weight = confidence * dot(normal, clusterNormal);

		directionalContribution[locId].v[directionIdx] = weight;
		sumWeights += weight;
		sumPoints += weight * point;
		sumNormals += weight * normal;
	}

	float oneOverWeight = 1 / sumWeights;

	outputPointCloud[locId] = sumPoints * oneOverWeight;
	outputNormals[locId] = Vector4f(sumNormals * oneOverWeight, 1);
	directionalContribution[locId] *= oneOverWeight;
}

template<bool useSmoothing, bool flipNormals>
_CPU_AND_GPU_CODE_ inline void processPixelICP(DEVICEPTR(Vector4f)* pointsMap, DEVICEPTR(Vector4f)* normalsMap,
                                               const CONSTPTR(Vector4f)* pointsRay, const CONSTPTR(Vector4f)* normalsRay,
                                               const THREADPTR(Vector2i)& imgSize,
                                               const THREADPTR(int)& x, const THREADPTR(int)& y, float voxelSize,
                                               const THREADPTR(Vector3f)& lightSource)
{
	Vector3f outNormal;
	float angle;

	int locId = x + y * imgSize.x;
	Vector4f point = pointsRay[locId];

	bool foundPoint = point.w > 0.0f;

	computeNormalAndAngle<useSmoothing, flipNormals>(pointsRay, normalsRay, lightSource, imgSize, x, y, foundPoint, outNormal, angle);

	if (foundPoint)
	{
		pointsMap[locId] = Vector4f(point.toVector3(), point.w);
		normalsMap[locId] = Vector4f(outNormal, 0);
	} else
	{
		Vector4f out4(0, 0, 0, -1);
		pointsMap[locId] = out4;
		normalsMap[locId] = out4;
	}
}

template<bool useSmoothing, bool flipNormals>
_CPU_AND_GPU_CODE_ inline void
processPixelGrey_ImageNormals(DEVICEPTR(Vector4u)* outRendering,
                              const CONSTPTR(Vector4f)* pointsRay, const CONSTPTR(Vector4f)* normalsRay,
                              const THREADPTR(Vector2i)& imgSize, const THREADPTR(int)& x, const THREADPTR(int)& y,
                              float voxelSize, const THREADPTR(Vector3f)& lightSource)
{
	Vector3f outNormal;
	float angle;

	int locId = x + y * imgSize.x;
	Vector4f point = pointsRay[locId];

	bool foundPoint = point.w > 0.0f;
	computeNormalAndAngle<useSmoothing, flipNormals>(pointsRay, normalsRay, lightSource, imgSize,
		x, y, foundPoint, outNormal, angle);

	if (foundPoint) drawPixelGrey(outRendering[locId], angle);
	else outRendering[locId] = Vector4u((uchar) 0);
}

template<bool useSmoothing, bool flipNormals>
_CPU_AND_GPU_CODE_ inline void
processPixelNormals_ImageNormals(DEVICEPTR(Vector4u)* outRendering, const CONSTPTR(Vector4f)* pointsRay, const CONSTPTR(Vector4f)* normalsRay,
                                 const THREADPTR(Vector2i)& imgSize, const THREADPTR(int)& x, const THREADPTR(int)& y,
                                 float voxelSize, Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	int locId = x + y * imgSize.x;
	Vector4f point = pointsRay[locId];

	bool foundPoint = point.w > 0.0f;
	computeNormalAndAngle<useSmoothing, flipNormals>(pointsRay, normalsRay, lightSource, imgSize,
	                                                 x, y, foundPoint, outNormal, angle);

	if (foundPoint) drawPixelNormal(outRendering[locId], outNormal);
	else outRendering[locId] = Vector4u((uchar) 0);
}

template<bool useSmoothing, bool flipNormals>
_CPU_AND_GPU_CODE_ inline void
processPixelConfidence_ImageNormals(DEVICEPTR(Vector4u)* outRendering, const CONSTPTR(Vector4f)* pointsRay,
                                    const CONSTPTR(Vector4f)* normalsRay,
                                    const THREADPTR(Vector2i)& imgSize, const THREADPTR(int)& x,
                                    const THREADPTR(int)& y, const ITMSceneParams& sceneParams, Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	int locId = x + y * imgSize.x;
	Vector4f point = pointsRay[locId];

	bool foundPoint = point.w > 0.0f;
	computeNormalAndAngle<useSmoothing, flipNormals>(pointsRay, normalsRay, lightSource, imgSize,
	                                                 x, y, foundPoint, outNormal, angle);

	if (foundPoint)
		drawPixelConfidence(outRendering[locId], angle,
		                    point.w / static_cast<float>(sceneParams.maxW));
	else outRendering[locId] = Vector4u((uchar) 0);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void processPixelGrey_SDFNormals(DEVICEPTR(Vector4u)& outRendering, const CONSTPTR(Vector3f)& point,
                                                           const Vector6f* directionalContribution,
                                                           bool foundPoint, const CONSTPTR(TVoxel)* voxelData,
                                                           const CONSTPTR(typename TIndex::IndexData)* voxelIndex,
                                                           Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	computeNormalAndAngle<TVoxel, TIndex>(foundPoint, point, directionalContribution, voxelData, voxelIndex, lightSource,
	                                      outNormal, angle);

	if (foundPoint) drawPixelGrey(outRendering, angle);
	else outRendering = Vector4u((uchar) 0);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void processPixelColour(
	DEVICEPTR(Vector4u)& outRendering, const CONSTPTR(Vector3f)& point,
	const Vector6f* directionalContribution, bool foundPoint, const CONSTPTR(TVoxel)* voxelData,
	const CONSTPTR(typename TIndex::IndexData)* voxelIndex, const Vector3f lightSource)
{
	float angle;
	Vector3f outNormal;
//	computeNormalAndAngle<TVoxel, TIndex>(foundPoint, point, directionalContribution, voxelData, voxelIndex,
//	                                      lightSource, outNormal, angle);
	angle =1;

	if (foundPoint)
	{
		if (directionalContribution)
			drawPixelColourDirectional<TVoxel, TIndex>(outRendering, point, *directionalContribution, angle, voxelData,
			                                           voxelIndex);
		else
			drawPixelColourDefault<TVoxel, TIndex>(outRendering, point, angle, voxelData, voxelIndex);
	} else outRendering = Vector4u((uchar) 0);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void processPixelDepth(
	DEVICEPTR(Vector4u)& outRendering, const CONSTPTR(Vector3f)& point, bool foundPoint,
	const Matrix4f& T_CW, const float voxelSize, const float maxDepth)
{
	Vector4f pointCamera = T_CW * Vector4f(point, 1);
	float depth = pointCamera.z;

	if (not foundPoint or depth <= 0.0f)
	{
		outRendering = Vector4u(0, 0, 0, 255);
		return;
	}

	float hue = (1 - (maxDepth - depth) / maxDepth) * 240;

	outRendering = Vector4f(HSVtoRGB(hue, 1, 1) * 255, 255).toUChar();
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void processPixelNormal_SDFNormals(DEVICEPTR(Vector4u)& outRendering, const CONSTPTR(Vector3f)& point,
                                                             const Vector6f* directionalContribution,
                                                             bool foundPoint, const CONSTPTR(TVoxel)* voxelData,
                                                             const CONSTPTR(typename TIndex::IndexData)* voxelIndex,
                                                             Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	computeNormalAndAngle<TVoxel, TIndex>(foundPoint, point, directionalContribution, voxelData, voxelIndex, lightSource,
	                                      outNormal, angle);

	if (foundPoint) drawPixelNormal(outRendering, outNormal);
	else outRendering = Vector4u((uchar) 0);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void
processPixelConfidence_SDFNormals(DEVICEPTR(Vector4u)& outRendering, const CONSTPTR(Vector4f)& point,
                                  const Vector6f* directionalContribution,
                                  bool foundPoint, const CONSTPTR(TVoxel)* voxelData,
                                  const CONSTPTR(typename TIndex::IndexData)* voxelIndex,
                                  const ITMSceneParams& sceneParams, Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	computeNormalAndAngle<TVoxel, TIndex>(foundPoint, TO_VECTOR3(point), directionalContribution, voxelData, voxelIndex,
	                                      lightSource, outNormal, angle);

	if (foundPoint)
		drawPixelConfidence(outRendering, angle,
		                    point.w / static_cast<float>(sceneParams.maxW));
	else outRendering = Vector4u((uchar) 0);
}

_CPU_AND_GPU_CODE_ inline void
processPixelError(Vector4u* outRendering, const Vector4f* pointsRay, const Vector4f* normalsRay,
                  const float* depth,
                  const Matrix4f& depthImageInvPose, const Matrix4f& sceneRenderingPose,
                  const Vector4f& intrinsics, const Vector2i& imgSize, const float maxError,
                  const int x, const int y)
{
	int locId = x + y * imgSize.width;

	float A[6];
	float b;
	bool isValidPoint = computePerPointGH_Depth_Ab<false, false>(A, b, x, y, depth[locId], imgSize, intrinsics, imgSize,
	                                                             intrinsics, depthImageInvPose, sceneRenderingPose,
	                                                             pointsRay, normalsRay, 100.0);
	float angle = -(sceneRenderingPose * normalsRay[locId]).z;

	if (!isValidPoint) {// or angle <= 0.0) {
		if (depth[locId] > 0)
			outRendering[locId] = Vector4u(127, 127, 127, 255);
		else if (pointsRay[locId].w > 0)
			outRendering[locId] = Vector4u(255, 255, 255, 255);
		else
			outRendering[locId] = Vector4u(0, 0, 0, 0);
		return;
	}

	b = MIN(fabs(b / maxError), 1); // normalize

	Vector4f color = Vector4f(HSVtoRGB((1 - b) * 240, 1, 1) * 255, 255);
//	color = (0.8f * angle + 0.2f) * color;
	outRendering[locId] = color.toUChar();
}

} // namespace ITMLib
