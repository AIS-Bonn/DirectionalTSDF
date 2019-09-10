// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <math_constants.h>
#include "ITMLib/Objects/Scene/ITMDirectional.h"
#include "ITMLib/Objects/Scene/ITMMultiSceneAccess.h"
#include "ITMLib/Objects/Scene/ITMRepresentationAccess.h"
#include "ITMLib/Engines/Reconstruction/Interface/ITMSceneReconstructionEngine.h"
#include "ITMLib/Utils/ITMProjectionUtils.h"

namespace ITMLib
{

static const CONSTPTR(int) MAX_RENDERING_BLOCKS = 65536 * 4;
//static const int MAX_RENDERING_BLOCKS = 16384;
static const CONSTPTR(int) minmaximg_subsample = 8;

#if !(defined __METALC__)

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

#endif


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
                                                     const Vector6f *directionalContribution,
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
				outNormal += directionalContribution->v[direction] * computeSingleNormalFromSDF(voxelBlockData, indexData, point, TSDFDirection(direction)).normalised();
		}
		outNormal = outNormal.normalised();
	}
	else
	{
		outNormal = computeSingleNormalFromSDF(voxelBlockData, indexData, point, TSDFDirection::NONE).normalised();
	}

	angle = outNormal.x * lightSource.x + outNormal.y * lightSource.y + outNormal.z * lightSource.z;
	if (!(angle > 0.0)) foundPoint = false;
}

template<bool useSmoothing, bool flipNormals>
_CPU_AND_GPU_CODE_ inline void
computeNormalAndAngle(THREADPTR(bool)& foundPoint, const THREADPTR(int)& x, const THREADPTR(int)& y,
                      const CONSTPTR(Vector4f)* pointsRay, const THREADPTR(Vector3f)& lightSource,
                      const THREADPTR(float)& voxelSize,
                      const THREADPTR(Vector2i)& imgSize, THREADPTR(Vector3f)& outNormal, THREADPTR(float)& angle)
{
	if (!foundPoint) return;

	Vector4f xp1_y, xm1_y, x_yp1, x_ym1;

	if (useSmoothing)
	{
		if (y <= 2 || y >= imgSize.y - 3 || x <= 2 || x >= imgSize.x - 3)
		{
			foundPoint = false;
			return;
		}

		xp1_y = pointsRay[(x + 2) + y * imgSize.x], x_yp1 = pointsRay[x + (y + 2) * imgSize.x];
		xm1_y = pointsRay[(x - 2) + y * imgSize.x], x_ym1 = pointsRay[x + (y - 2) * imgSize.x];
	} else
	{
		if (y <= 1 || y >= imgSize.y - 2 || x <= 1 || x >= imgSize.x - 2)
		{
			foundPoint = false;
			return;
		}

		xp1_y = pointsRay[(x + 1) + y * imgSize.x], x_yp1 = pointsRay[x + (y + 1) * imgSize.x];
		xm1_y = pointsRay[(x - 1) + y * imgSize.x], x_ym1 = pointsRay[x + (y - 1) * imgSize.x];
	}

	Vector4f diff_x(0.0f, 0.0f, 0.0f, 0.0f), diff_y(0.0f, 0.0f, 0.0f, 0.0f);

	bool doPlus1 = false;
	if (xp1_y.w <= 0 || x_yp1.w <= 0 || xm1_y.w <= 0 || x_ym1.w <= 0) doPlus1 = true;
	else
	{
		diff_x = xp1_y - xm1_y, diff_y = x_yp1 - x_ym1;

		float length_diff = MAX(diff_x.x * diff_x.x + diff_x.y * diff_x.y + diff_x.z * diff_x.z,
		                        diff_y.x * diff_y.x + diff_y.y * diff_y.y + diff_y.z * diff_y.z);

		if (length_diff * voxelSize * voxelSize > (0.15f * 0.15f)) doPlus1 = true;
	}

	if (doPlus1)
	{
		if (useSmoothing)
		{
			xp1_y = pointsRay[(x + 1) + y * imgSize.x];
			x_yp1 = pointsRay[x + (y + 1) * imgSize.x];
			xm1_y = pointsRay[(x - 1) + y * imgSize.x];
			x_ym1 = pointsRay[x + (y - 1) * imgSize.x];
			diff_x = xp1_y - xm1_y;
			diff_y = x_yp1 - x_ym1;
		}

		if (xp1_y.w <= 0 || x_yp1.w <= 0 || xm1_y.w <= 0 || x_ym1.w <= 0)
		{
			foundPoint = false;
			return;
		}
	}

	outNormal.x = -(diff_x.y * diff_y.z - diff_x.z * diff_y.y);
	outNormal.y = -(diff_x.z * diff_y.x - diff_x.x * diff_y.z);
	outNormal.z = -(diff_x.x * diff_y.y - diff_x.y * diff_y.x);

	if (flipNormals) outNormal = -outNormal;

	outNormal = outNormal.normalised();

	angle = outNormal.x * lightSource.x + outNormal.y * lightSource.y + outNormal.z * lightSource.z;
	if (!(angle > 0.0)) foundPoint = false;
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

_CPU_AND_GPU_CODE_ inline void
drawPixelConfidence(DEVICEPTR(Vector4u)& dest, const THREADPTR(float)& angle, const THREADPTR(float)& confidence)
{
	//Vector4f color_red(255, 0, 0, 255), color_green(0, 255, 0, 255);
	float confidenceNorm = CLAMP(confidence, 0, 100.f) / 100.0f;

	Vector4f color;
	color.r = (uchar) (baseCol(confidenceNorm) * 255.0f);
	color.g = (uchar) (baseCol(confidenceNorm - 0.5f) * 255.0f);
	color.b = (uchar) (baseCol(confidenceNorm + 0.5f) * 255.0f);
	color.a = 255;

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
	DEVICEPTR(Vector4u)& dest, const CONSTPTR(Vector3f)& point, const Vector6f &directionalContribution,
	const CONSTPTR(TVoxel)* voxelBlockData, const CONSTPTR(typename TIndex::IndexData)* indexData)
{
	dest.x = 0;
	dest.y = 0;
	dest.z = 0;

	const Vector3f directionColors[6] = {
		Vector3f(1, 0, 0),
		Vector3f(0, 1, 0),
		Vector3f(1, 1, 0),
		Vector3f(0, 1, 1),
		Vector3f(1, 0, 1),
		Vector3f(1, 1, 0)
	};

	// Debugging: color in contributing directions
	for (TSDFDirection_type direction = 0; direction < N_DIRECTIONS; direction++)
	{
		const Vector3f &clr = directionColors[direction];
		dest.x += (uchar) (clr.x * 255.0f * directionalContribution[direction]);
		dest.y += (uchar) (clr.y * 255.0f * directionalContribution[direction]);
		dest.z += (uchar) (clr.z * 255.0f * directionalContribution[direction]);
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
	dest.w = 255;
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void drawPixelColourDefault(DEVICEPTR(Vector4u)& dest, const CONSTPTR(Vector3f)& point,
                                                      const CONSTPTR(TVoxel)* voxelBlockData,
                                                      const CONSTPTR(typename TIndex::IndexData)* indexData)
{
	Vector4f clr = VoxelColorReader<TVoxel::hasColorInformation, TVoxel, TIndex>::interpolate(voxelBlockData, indexData,
	                                                                                          point, TSDFDirection::NONE);

	dest.x = (uchar) (clr.x * 255.0f);
	dest.y = (uchar) (clr.y * 255.0f);
	dest.z = (uchar) (clr.z * 255.0f);
	dest.w = 255;
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline bool castRayDefault(DEVICEPTR(Vector4f)& pt_out,
                                              float& distance_out,
                                              DEVICEPTR(HashEntryVisibilityType)* entriesVisibleType,
                                              int x, int y, const CONSTPTR(TVoxel)* voxelData,
                                              const CONSTPTR(typename TIndex::IndexData)* voxelIndex,
                                              Matrix4f invM, Vector4f invProjParams, float oneOverVoxelSize, float mu,
                                              const CONSTPTR(Vector2f)& viewFrustum_minmax,
                                              const TSDFDirection direction=TSDFDirection::NONE,
                                              float maxDistance=-1)
{
	Vector4f pt_camera_f; Vector3f pt_block_s, pt_block_e, rayDirection, pt_result;
	int vmIndex;
	float sdfValue = 1.0f, confidence = 0;
	float totalLength, stepLength, totalLengthMax, stepScale;

	pt_out = Vector4f(0, 0, 0, 0);

	stepScale = mu * oneOverVoxelSize;

	pt_camera_f = Vector4f(reprojectImagePoint(x, y, viewFrustum_minmax.x, invProjParams), 1.0f);
	totalLength = length(TO_VECTOR3(pt_camera_f)) * oneOverVoxelSize;
	pt_block_s = TO_VECTOR3(invM * pt_camera_f) * oneOverVoxelSize;

	pt_camera_f = Vector4f(reprojectImagePoint(x, y,
		maxDistance > 0 ? maxDistance : viewFrustum_minmax.y,
		invProjParams), 1.0f);
	totalLengthMax = length(TO_VECTOR3(pt_camera_f)) * oneOverVoxelSize;
	pt_block_e = TO_VECTOR3(invM * pt_camera_f) * oneOverVoxelSize;

	rayDirection = pt_block_e - pt_block_s;
	float direction_norm = 1.0f / sqrt(rayDirection.x * rayDirection.x + rayDirection.y * rayDirection.y + rayDirection.z * rayDirection.z);
	rayDirection *= direction_norm;

	pt_result = pt_block_s;

	typename TIndex::IndexCache cache;

	float lastSDFValue = sdfValue;
	while (totalLength < totalLengthMax) {
		sdfValue = readFromSDF_float_uninterpolated(voxelData, voxelIndex, pt_result, direction, vmIndex, cache);

		if (entriesVisibleType)
		{
			if (vmIndex) entriesVisibleType[vmIndex - 1] = VISIBLE_IN_MEMORY;
		}

		if (!vmIndex) {
			stepLength = SDF_BLOCK_SIZE;
		} else {
			if ((sdfValue <= 0.1f) && (sdfValue >= -0.5f)) {
				sdfValue = readFromSDF_float_interpolated(voxelData, voxelIndex, pt_result, direction, vmIndex, cache);
			}
			if (lastSDFValue > 0 and sdfValue <= 0)// and lastSDFValue - sdfValue < 1.1)
			{
				break;
			}
			stepLength = MAX(sdfValue * stepScale, 1.0f);
		}

		lastSDFValue = sdfValue;
		pt_result += stepLength * rayDirection; totalLength += stepLength;
	}

	lastSDFValue = sdfValue;
	// Perform 2 additional steps to get more accurate zero crossing
	if (sdfValue > 0.0f)
		return false;

	for (int i = 0; i < 5 and fabs(sdfValue) > 1e-6; i++)
	{
		stepLength = sdfValue * stepScale;
		pt_result += stepLength * rayDirection;
		totalLength += stepLength;

		sdfValue = readWithConfidenceFromSDF_float_interpolated(confidence, voxelData, voxelIndex, pt_result, direction, vmIndex, cache);
	}

	if (fabs(lastSDFValue) < fabs(sdfValue) or fabs(sdfValue) > 0.1)
		return false;

	distance_out = totalLength / oneOverVoxelSize;
	pt_out = Vector4f(pt_result, confidence + 1.0f);

	return true;
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline bool castRayDirectional2(DEVICEPTR(Vector4f) &pt_out, Vector6f *directionalContribution,
                                                  DEVICEPTR(HashEntryVisibilityType) *entriesVisibleType,
                                                  int x, int y, const CONSTPTR(TVoxel) *voxelData,
                                                  const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
                                                  Matrix4f invM, Vector4f invProjParams, float oneOverVoxelSize, float mu, const CONSTPTR(Vector2f) & viewFrustum_minmax)
{
	Vector4f pt_camera_f; Vector3f pt_block_s, pt_block_e, rayDirection, pt_result;
	float totalLength, totalLengthMax, stepScale;

	pt_out = Vector4f(0, 0, 0, 0);

	stepScale = mu * oneOverVoxelSize;

	pt_camera_f = Vector4f(reprojectImagePoint(x, y, viewFrustum_minmax.x, invProjParams), 1.0f);
	totalLength = length(TO_VECTOR3(pt_camera_f)) * oneOverVoxelSize;
	pt_block_s = TO_VECTOR3(invM * pt_camera_f) * oneOverVoxelSize;

	pt_camera_f = Vector4f(reprojectImagePoint(x, y,
	                                           viewFrustum_minmax.y,
	                                           invProjParams), 1.0f);
	totalLengthMax = length(TO_VECTOR3(pt_camera_f)) * oneOverVoxelSize;
	pt_block_e = TO_VECTOR3(invM * pt_camera_f) * oneOverVoxelSize;

	rayDirection = pt_block_e - pt_block_s;
	float direction_norm = 1.0f / sqrt(rayDirection.x * rayDirection.x + rayDirection.y * rayDirection.y + rayDirection.z * rayDirection.z);
	rayDirection *= direction_norm;

	pt_result = pt_block_s;

	/// Find first positive zero crossing
	int vmIndex;
	float sdfValues[N_DIRECTIONS] = {1, 1, 1, 1, 1, 1};
	float lastSDFValues[N_DIRECTIONS] = {1, 1, 1, 1, 1, 1};
	typename TIndex::IndexCache cache[N_DIRECTIONS];
	float stepLength = SDF_BLOCK_SIZE;
	TSDFDirection_type firstCrossingDirection = static_cast<TSDFDirection_type>(TSDFDirection::NONE);
	while (totalLength < totalLengthMax) {
		bool foundZeroCrossing = false;
		int numReadableValues = 0;
		for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
		{
			const auto direction = TSDFDirection(directionIdx);

			sdfValues[directionIdx] = readFromSDF_float_uninterpolated(voxelData, voxelIndex, pt_result, direction, vmIndex, cache[directionIdx]);

			if (vmIndex) {
				numReadableValues++;
				float confidence = 0;
				if (sdfValues[directionIdx] <= 0.5 and sdfValues[directionIdx] >= -0.5) {
					sdfValues[directionIdx] = readWithConfidenceFromSDF_float_interpolated(confidence, voxelData, voxelIndex, pt_result,
					                                                           TSDFDirection(directionIdx), vmIndex, cache[directionIdx]);
				}
				if (lastSDFValues[directionIdx] > 0 and sdfValues[directionIdx] <= 0)
				{
					totalLengthMax = MIN(totalLengthMax, totalLength + SDF_BLOCK_SIZE);
					if (confidence > 0)
					{
						firstCrossingDirection = directionIdx;
						foundZeroCrossing = true;
						break;
					}
				}
				stepLength = MIN(stepLength, MAX(sdfValues[directionIdx] * stepScale, 1.0f));
			}
			lastSDFValues[directionIdx] = sdfValues[directionIdx];
		}

		pt_result += stepLength * rayDirection;
		totalLength += stepLength;

		if (numReadableValues == 0)
			stepLength = SDF_BLOCK_SIZE;

		if (foundZeroCrossing)
			break;
	}

	if (firstCrossingDirection >= N_DIRECTIONS or sdfValues[firstCrossingDirection] > 0.0)
		return false;

	/// Check which directions have usable information
	float weights[N_DIRECTIONS] = {1, 1, 1, 1, 1, 1};
	Vector3f gradients[N_DIRECTIONS];
	float maxConfidence = 0;
	TSDFDirection_type maxConfidenceIdx = static_cast<TSDFDirection_type>(TSDFDirection::NONE);
	for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
	{
		float confidence = 0;
		sdfValues[directionIdx] = readWithConfidenceFromSDF_float_interpolated(confidence, voxelData, voxelIndex, pt_result,
		                                                                       TSDFDirection(directionIdx), vmIndex, cache[directionIdx]);
		if (confidence == 0)
		{
			weights[directionIdx] = 0;
			continue;
		}
		gradients[directionIdx] = computeSingleNormalFromSDF(voxelData, voxelIndex, pt_result,
			TSDFDirection(directionIdx)).normalised();
		confidence *= DirectionWeight(gradients[directionIdx], TSDFDirection(directionIdx));
		confidence *= SIGN(dot(gradients[directionIdx], -rayDirection));

		if (confidence > maxConfidence)
		{
			maxConfidence = confidence;
			maxConfidenceIdx = directionIdx;
		}
	}

	firstCrossingDirection = maxConfidenceIdx;

	if (firstCrossingDirection >= N_DIRECTIONS)
		return false;

	/// Perform steps to get closer to zero crossing
	float sdfValue = sdfValues[firstCrossingDirection];

	/// Determine directions and weights to use for final interpolation (surface gradient)
	float weights_[N_DIRECTIONS];
	ComputeDirectionWeights(gradients[firstCrossingDirection], weights_);

	for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
	{
		if (weights[directionIdx] <= 0 or directionIdx == firstCrossingDirection)
			continue;

		if (weights_[directionIdx] < direction_weight_threshold)
		{
			weights[directionIdx] = 0;
			continue;
		}

		weights[directionIdx] = weights_[directionIdx] * dot(gradients[directionIdx], gradients[firstCrossingDirection]);
		if (weights[directionIdx] != weights[directionIdx])
			weights[directionIdx] = 0; // check NaN
	}

	stepScale =  mu * oneOverVoxelSize / MAX(fabs(dot(gradients[firstCrossingDirection], rayDirection)), 0.5); // Adapt step scale to angle between surface and observation ray

	float dbg_[10];
	float dbg2_[10];

	/// Perform additional steps to get more accurate zero crossing (interpolate directions)
	float contributionWeights[N_DIRECTIONS] = {0, 0, 0, 0, 0, 0};
	float lastSDFValue;
	float weightSum = 0;
	for (int i = 0; i < 5 and fabs(sdfValue) > 1e-6; i++)
	{
		lastSDFValue = sdfValue;
		sdfValue = 0;
    weightSum = 0;
		for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
		{
			if (weights[directionIdx] < direction_weight_threshold) continue;

			float confidence = 0;
			float value = readWithConfidenceFromSDF_float_interpolated(confidence, voxelData, voxelIndex, pt_result,
				TSDFDirection(directionIdx), vmIndex, cache[directionIdx]);

			if (confidence == 0)
			{
				weights[directionIdx] = 0; // Direction not applicable for estimation
				continue;
			}

			contributionWeights[directionIdx] = confidence * weights[directionIdx];

			sdfValue += contributionWeights[directionIdx] * value;
			weightSum += contributionWeights[directionIdx];
		}
		if (weightSum == 0)
			return false;

		sdfValue /= weightSum;
		if (SIGN(sdfValue) != SIGN(lastSDFValue) and (fabs(sdfValue) - fabs(lastSDFValue)) / fabs(lastSDFValue) > -0.75)
		{ // Sign hopping with little to no reduction in magnitude
			stepScale *= 0.5;
		}
		dbg2_[i] = stepScale;

		stepLength = sdfValue * stepScale;
		pt_result += stepLength * rayDirection;
		totalLength += stepLength;

		dbg_[i] = sdfValue;
	}
	if (fabs(sdfValue) > 0.1)
		return false;

	if (directionalContribution)
	{
		for (int i = 0; i < 6; i++)
		{
			directionalContribution->v[i] = MAX(contributionWeights[i] / weightSum, 0);
		}
	}

	pt_out = Vector4f(pt_result, weightSum + 1.0f);
	return true;
}

/**
 *
 * @param directionalContribution vector of floats that holds the proportional contribution of every direction to the pixel
 */
template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline bool castRayDirectional(DEVICEPTR(Vector4f) &pt_out, Vector6f *directionalContribution,
	                                                DEVICEPTR(HashEntryVisibilityType) *entriesVisibleType,
                                                  int x, int y, const CONSTPTR(TVoxel) *voxelData,
                                                  const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
                                                  Matrix4f invM, Vector4f invProjParams, float oneOverVoxelSize, float mu, const CONSTPTR(Vector2f) & viewFrustum_minmax)
{
	Vector3f ptStart_camera = reprojectImagePoint(x, y, viewFrustum_minmax.x, invProjParams);
	Vector3f ptStart_world = (invM * Vector4f(ptStart_camera, 1.0f)).toVector3() * oneOverVoxelSize;

	Vector3f ptEnd_camera = reprojectImagePoint(x, y, viewFrustum_minmax.y, invProjParams);
	Vector3f ptEnd_world = (invM * Vector4f(ptEnd_camera, 1.0f)).toVector3() * oneOverVoxelSize;

	Vector3f rayDirection_world = (ptEnd_world - ptStart_world).normalised();

	/// collect TSDF directions which are applicable for ray
	float directionWeights[N_DIRECTIONS];
	ComputeDirectionWeights(-rayDirection_world, directionWeights);

	float max = 0;
	int maxIdx = -1;
	for (uchar directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
	{
		if (directionWeights[directionIdx] < direction_weight_threshold)
			continue;
		if (directionWeights[directionIdx] > max)
		{
			max = directionWeights[directionIdx];
			maxIdx = directionIdx;
		}
	}
	int opositeIdx = 2 * (maxIdx / 2) + (1 - maxIdx % 2);

	const float MAX_LENGTH = 1e9;
	float directionIntersectionLength[N_DIRECTIONS] = {MAX_LENGTH, MAX_LENGTH, MAX_LENGTH, MAX_LENGTH, MAX_LENGTH, MAX_LENGTH};
	float directionIntersectionWeight[N_DIRECTIONS] = {0, 0, 0, 0, 0, 0};
	Vector3f directionIntersectionPoint[N_DIRECTIONS];
	Vector3f directionIntersectionNormal[N_DIRECTIONS];
	float maxIntersectionLength = -1;
	for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
	{
		if (directionIdx == opositeIdx)
			continue;
		Vector4f point;
		float distance = -1;
		if (castRayDefault<TVoxel, TIndex>(
				point, distance,
				entriesVisibleType, x, y, voxelData, voxelIndex,
				invM, invProjParams, oneOverVoxelSize, mu, viewFrustum_minmax, TSDFDirection(directionIdx),
				maxIntersectionLength > 0 ? maxIntersectionLength + SDF_BLOCK_SIZE / oneOverVoxelSize : -1))
		{
			maxIntersectionLength = MAX(maxIntersectionLength, distance);
#if 1
			const auto direction = TSDFDirection(directionIdx);
			directionIntersectionNormal[directionIdx] = computeSingleNormalFromSDF(
				voxelData, voxelIndex, point.toVector3(), direction).normalised();
			float directionCompliance = DirectionWeight(directionIntersectionNormal[directionIdx], direction);
			if (directionCompliance >= direction_weight_threshold)
			{
				directionIntersectionPoint[directionIdx] = point.toVector3();
				directionIntersectionLength[directionIdx] = distance;
				directionIntersectionWeight[directionIdx] = point.w - 1;
			}
#else
			directionIntersectionPoint[directionIdx] = point.toVector3();
			directionIntersectionLength[directionIdx] = distance;
			directionIntersectionWeight[directionIdx] = point.w - 1;
#endif
		}
	}

	/// Find closest intersection
	float currentIntersectionLength = MAX_LENGTH;
	int currentIntersectionIdx = -1;
	for (TSDFDirection_type idx = 0; idx < N_DIRECTIONS; idx++)
	{
		if (directionIntersectionLength[idx] < currentIntersectionLength)
		{
			currentIntersectionLength = directionIntersectionLength[idx];
			currentIntersectionIdx = idx;
		}
	}

	/// Find most suitable intersection (innermost one in vicinity of first intersection)
	float shortestIntersectionLength = currentIntersectionLength;
	int shortestIntersectionIdx = currentIntersectionIdx;
	for (TSDFDirection_type idx = 0; idx < N_DIRECTIONS; idx++)
	{
		if (directionIntersectionLength[idx] - shortestIntersectionLength > 0 and
		directionIntersectionLength[idx] - currentIntersectionLength < SDF_BLOCK_SIZE)
		{
			shortestIntersectionLength = directionIntersectionLength[idx];
			shortestIntersectionIdx = idx;
		}
	}
	currentIntersectionIdx = shortestIntersectionIdx;
	currentIntersectionLength = shortestIntersectionLength;

	float weightSum = 0;
	pt_out = Vector4f(0, 0, 0, 0);
	for (TSDFDirection_type idx = 0; idx < N_DIRECTIONS; idx++)
	{
		if (directionIntersectionLength[idx] - currentIntersectionLength < SDF_BLOCK_SIZE)
		{
			pt_out += Vector4f(directionIntersectionPoint[idx] * directionIntersectionWeight[idx], 0);
			weightSum += directionIntersectionWeight[idx];
		} else
		{
			directionIntersectionWeight[idx] = 0; // Set 0 to exclude from coloring
		}
	}
	float oneOverWeightSum = 1 / weightSum;

	pt_out *= oneOverWeightSum;
	pt_out.w = weightSum + 1.0f;

	if (currentIntersectionIdx < 0 or weightSum <= 0)
	{
		pt_out.w = 0.0f;
		return false;
	}

	if (directionalContribution)
	{
		for (int i = 0; i < 6; i++)
		{
			directionalContribution->v[i] = directionIntersectionWeight[i] * oneOverWeightSum;
		}
	}

	return true;
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline bool castRay(DEVICEPTR(Vector4f) &pt_out, Vector6f *directionalContribution, DEVICEPTR(HashEntryVisibilityType) *entriesVisibleType,
                                       int x, int y, const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
                                       Matrix4f invM, Vector4f invProjParams, float oneOverVoxelSize, float mu, const CONSTPTR(Vector2f) & viewFrustum_minmax,
                                       bool directionalTSDF)
{
	float distance;
	if (directionalTSDF)
//		return castRayDirectional<TVoxel, TIndex>(pt_out, directionalContribution, entriesVisibleType, x, y, voxelData, voxelIndex, invM, invProjParams, oneOverVoxelSize, mu, viewFrustum_minmax);
		return castRayDirectional2<TVoxel, TIndex>(pt_out, directionalContribution, entriesVisibleType, x, y, voxelData, voxelIndex, invM, invProjParams, oneOverVoxelSize, mu, viewFrustum_minmax);
//		return castRayDefault<TVoxel, TIndex>(pt_out, distance, entriesVisibleType, x, y, voxelData, voxelIndex, invM, invProjParams,
//																					oneOverVoxelSize, mu, viewFrustum_minmax, TSDFDirection::Z_NEG);
	else
		return castRayDefault<TVoxel, TIndex>(pt_out, distance, entriesVisibleType, x, y, voxelData, voxelIndex, invM, invProjParams,
	                                      oneOverVoxelSize, mu, viewFrustum_minmax);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void processPixelICP(DEVICEPTR(Vector4f) &pointsMap, DEVICEPTR(Vector4f) &normalsMap,
	const THREADPTR(Vector3f) & point, bool foundPoint, const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
	float voxelSize, const THREADPTR(Vector3f) &lightSource)
{
	Vector3f outNormal;
	float angle;

	computeNormalAndAngle<TVoxel, TIndex>(foundPoint, point, voxelData, voxelIndex, lightSource, outNormal, angle);

	if (foundPoint)
	{
		Vector4f outPoint4;
		outPoint4.x = point.x * voxelSize; outPoint4.y = point.y * voxelSize;
		outPoint4.z = point.z * voxelSize; outPoint4.w = 1.0f;
		pointsMap = outPoint4;

		Vector4f outNormal4;
		outNormal4.x = outNormal.x; outNormal4.y = outNormal.y; outNormal4.z = outNormal.z; outNormal4.w = 0.0f;
		normalsMap = outNormal4;
	}
	else
	{
		Vector4f out4;
		out4.x = 0.0f; out4.y = 0.0f; out4.z = 0.0f; out4.w = -1.0f;

		pointsMap = out4; normalsMap = out4;
	}
}

template<typename T>
_CPU_AND_GPU_CODE_
inline void insertionSort(const T *in, T *out, size_t *outIds, size_t length)
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
	int *clusterOutput, Vector3f &clusterNormalOutput)
{
	float distancesSorted[N_DIRECTIONS];
	size_t distancesSortedIds[N_DIRECTIONS];
	insertionSort(distances, distancesSorted, distancesSortedIds, N_DIRECTIONS);

	/// Cluster the distances. Number indicates starting index of a cluster (end indicated by next cluster or -1)
	int clusters[N_DIRECTIONS] = {-1, -1, -1, -1, -1, -1};
	float clusterConfidences[N_DIRECTIONS] = {0, 0, 0, 0, 0, 0};
	Vector3f clusterNormals[N_DIRECTIONS];
	uchar numClusters = 0;
	float lastDistance = -1;
	float interClusterMaxConfidence = -1;
	float sumConfidences = 0;
	for (uchar idx = 0; idx < N_DIRECTIONS and distancesSorted[idx] >= 0; idx++)
	{
		if (fabs(distancesSorted[idx] - lastDistance) > 2) // maximum distance in voxels
		{
			clusters[numClusters] = idx;
			numClusters++;
			interClusterMaxConfidence = -1;
		}
		lastDistance = distancesSorted[idx];

		float confidence = confidences[distancesSortedIds[idx]];
		clusterConfidences[numClusters - 1] += confidence;

		if (confidence > interClusterMaxConfidence)
		{
			clusterNormals[numClusters -1] = normals[distancesSortedIds[idx]];
			interClusterMaxConfidence = confidence;
		}

		sumConfidences += confidence;
	}

	int bestCluster = -1;
	for (uchar idx = 0; idx < N_DIRECTIONS and clusters[idx] >= 0; idx++)
	{
		if (clusterConfidences[idx] >= 0.1 * sumConfidences)
		{
			bestCluster = idx;
			break;
		}
	}

	for (int i = 0; i < N_DIRECTIONS; i++)
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

struct InputPointClouds{
	Vector4f *pointCloud[N_DIRECTIONS];
};

template<bool useSmoothing, bool flipNormals>
_CPU_AND_GPU_CODE_ inline void combineDirectionalPointClouds(
	DEVICEPTR(Vector4f) *outputPointCloud,
  const InputPointClouds &inputPointClouds,
	DEVICEPTR(Vector6f) *directionalContribution,
  const THREADPTR(Vector2i) &imgSize,
  const Matrix4f &invM, const Vector4f &invProjParams,
  const THREADPTR(int) &x, const THREADPTR(int) &y, float voxelSize)
{
	int locId = x + y * imgSize.x;

	Vector4f points[N_DIRECTIONS];
	Vector3f normals[N_DIRECTIONS];
	float confidences[N_DIRECTIONS] = {0, 0, 0, 0, 0, 0};

	float distances[N_DIRECTIONS] = {INF_FLOAT, INF_FLOAT, INF_FLOAT, INF_FLOAT, INF_FLOAT, INF_FLOAT};

	Vector3f viewRay = (invM * Vector4f(reprojectImagePoint(x, y, 1, invProjParams), 0)).toVector3().normalised();
	Vector3f cameraPos_world = (invM * Vector4f(0, 0, 0, 1)).toVector3();

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
		confidence = point.w;

		bool foundPoint = confidence > 0.0f;
		if (not foundPoint)
			continue;

		distance = length(cameraPos_world - point.toVector3());

		const Vector3f TSDFDirectionVectors[N_DIRECTIONS] = {
			Vector3f(1,  0,  0),
			Vector3f(-1, 0,  0),
			Vector3f(0,  1,  0),
			Vector3f(0,  -1, 0),
			Vector3f(0,  0,  1),
			Vector3f(0,  0,  -1)
		};
		float angle;
		computeNormalAndAngle<useSmoothing, flipNormals>(foundPoint, x, y, inputPointClouds.pointCloud[directionIdx],
		                                                 TSDFDirectionVectors[directionIdx], voxelSize, imgSize, normal, angle);

		if (not foundPoint)// or angle < direction_weight_threshold)
		{
			confidence = 0;
			continue;
		}

		// Compliance of surface gradient wrt direction
		confidence *= angle; //DirectionWeight(normal, TSDFDirection(directionIdx));
		// Exclude points on opposite side
		confidence *= SIGN(dot(normal, -viewRay));

		sumConfidences += confidence > 0 ? confidence : 0;
	}

	int cluster[N_DIRECTIONS];
	Vector3f clusterNormal;
	findBestCluster(distances, confidences, normals, cluster, clusterNormal);

	if (cluster[0] < 0)
	{
		outputPointCloud[locId] = Vector4f(0, 0, 0, 0);
		return;
	}

	float sumWeights = 0;
	Vector4f sumPoints(0, 0, 0, 0);
	for (uchar idx = 0; idx < N_DIRECTIONS and cluster[idx] >= 0; idx++)
	{
		TSDFDirection_type directionIdx = cluster[idx];

		Vector4f &point = points[directionIdx];
		Vector3f &normal = normals[directionIdx];
		float &confidence = confidences[directionIdx];

		if (confidence <= 0)
			continue;

		// Filter directions, that don't apply to gradient
		if (DirectionWeight(clusterNormal, TSDFDirection(directionIdx)) < direction_weight_threshold)
			continue;

		// Filter gradients, that are too different
		if (dot(normal, clusterNormal) < direction_weight_threshold)
			continue;

		float weight = confidence * dot(normal, clusterNormal);

		directionalContribution[locId].v[directionIdx] = weight;
		sumWeights += weight;
		sumPoints += weight * point;
	}
	float oneOverWeight = 1 / sumWeights;

	outputPointCloud[locId] = sumPoints * oneOverWeight;
	for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
	{
		directionalContribution[locId].v[directionIdx] *= oneOverWeight;
	}
}

template<bool useSmoothing, bool flipNormals>
_CPU_AND_GPU_CODE_ inline void processPixelICP(DEVICEPTR(Vector4f) *pointsMap, DEVICEPTR(Vector4f) *normalsMap,
	const CONSTPTR(Vector4f) *pointsRay, const THREADPTR(Vector2i) &imgSize, const THREADPTR(int) &x, const THREADPTR(int) &y, float voxelSize,
	const THREADPTR(Vector3f) &lightSource)
{
	Vector3f outNormal;
	float angle;

	int locId = x + y * imgSize.x;
	Vector4f point = pointsRay[locId];

	bool foundPoint = point.w > 0.0f;

	computeNormalAndAngle<useSmoothing, flipNormals>(foundPoint, x, y, pointsRay, lightSource, voxelSize, imgSize, outNormal, angle);

	if (foundPoint)
	{
		Vector4f outPoint4;
		outPoint4.x = point.x * voxelSize; outPoint4.y = point.y * voxelSize;
		outPoint4.z = point.z * voxelSize; outPoint4.w = point.w;//outPoint4.w = 1.0f;
		pointsMap[locId] = outPoint4;

		Vector4f outNormal4;
		outNormal4.x = outNormal.x; outNormal4.y = outNormal.y; outNormal4.z = outNormal.z; outNormal4.w = 0.0f;
		normalsMap[locId] = outNormal4;
	}
	else
	{
		Vector4f out4;
		out4.x = 0.0f; out4.y = 0.0f; out4.z = 0.0f; out4.w = -1.0f;

		pointsMap[locId] = out4; normalsMap[locId] = out4;
	}
}

template<bool useSmoothing, bool flipNormals>
_CPU_AND_GPU_CODE_ inline void processPixelGrey_ImageNormals(DEVICEPTR(Vector4u) *outRendering, const CONSTPTR(Vector4f) *pointsRay, 
	const THREADPTR(Vector2i) &imgSize, const THREADPTR(int) &x, const THREADPTR(int) &y, float voxelSize, const THREADPTR(Vector3f) &lightSource)
{
	Vector3f outNormal;
	float angle;

	int locId = x + y * imgSize.x;
	Vector4f point = pointsRay[locId];

	bool foundPoint = point.w > 0.0f;
	computeNormalAndAngle<useSmoothing, flipNormals>(foundPoint, x, y, pointsRay, lightSource, voxelSize, imgSize, outNormal, angle);

	if (foundPoint) drawPixelGrey(outRendering[locId], angle);
	else outRendering[locId] = Vector4u((uchar)0);
}

template<bool useSmoothing, bool flipNormals>
_CPU_AND_GPU_CODE_ inline void processPixelNormals_ImageNormals(DEVICEPTR(Vector4u) *outRendering, const CONSTPTR(Vector4f) *pointsRay,
	const THREADPTR(Vector2i) &imgSize, const THREADPTR(int) &x, const THREADPTR(int) &y, float voxelSize, Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	int locId = x + y * imgSize.x;
	Vector4f point = pointsRay[locId];

	bool foundPoint = point.w > 0.0f;
	computeNormalAndAngle<useSmoothing, flipNormals>(foundPoint, x, y, pointsRay, lightSource, voxelSize, imgSize, outNormal, angle);

	if (foundPoint) drawPixelNormal(outRendering[locId], outNormal);
	else outRendering[locId] = Vector4u((uchar)0);
}

template<bool useSmoothing, bool flipNormals>
_CPU_AND_GPU_CODE_ inline void processPixelConfidence_ImageNormals(DEVICEPTR(Vector4u) *outRendering, const CONSTPTR(Vector4f) *pointsRay,
	const THREADPTR(Vector2i) &imgSize, const THREADPTR(int) &x, const THREADPTR(int) &y, float voxelSize, Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	int locId = x + y * imgSize.x;
	Vector4f point = pointsRay[locId];

	bool foundPoint = point.w > 0.0f;
	computeNormalAndAngle<useSmoothing, flipNormals>(foundPoint, x, y, pointsRay, lightSource, voxelSize, imgSize, outNormal, angle);

	if (foundPoint) drawPixelConfidence(outRendering[locId], angle, point.w - 1.0f);
	else outRendering[locId] = Vector4u((uchar)0);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void processPixelGrey(DEVICEPTR(Vector4u) &outRendering, const CONSTPTR(Vector3f) & point,
	const Vector6f *directionalContribution,
	bool foundPoint, const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex, 
	Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	computeNormalAndAngle<TVoxel, TIndex>(foundPoint, point, directionalContribution, voxelData, voxelIndex, lightSource, outNormal, angle);

	if (foundPoint) drawPixelGrey(outRendering, angle);
	else outRendering = Vector4u((uchar)0);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void processPixelColourDirectional(
	DEVICEPTR(Vector4u) &outRendering, const CONSTPTR(Vector3f) &point, const Vector6f &directionalContribution,
	bool foundPoint, const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex)
{
	if (foundPoint) drawPixelColourDirectional<TVoxel, TIndex>(outRendering, point, directionalContribution, voxelData, voxelIndex);
	else outRendering = Vector4u((uchar)0);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void processPixelColour(
	DEVICEPTR(Vector4u) &outRendering, const CONSTPTR(Vector3f) & point,
	const Vector6f *directionalContribution, bool foundPoint, const CONSTPTR(TVoxel) *voxelData,
	const CONSTPTR(typename TIndex::IndexData) *voxelIndex)
{
	if (foundPoint)
	{
		if (directionalContribution)
			drawPixelColourDirectional<TVoxel, TIndex>(outRendering, point, *directionalContribution, voxelData, voxelIndex);
		else
			drawPixelColourDefault<TVoxel, TIndex>(outRendering, point, voxelData, voxelIndex);
	}
	else outRendering = Vector4u((uchar)0);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void processPixelNormal(DEVICEPTR(Vector4u) &outRendering, const CONSTPTR(Vector3f) & point,
	const Vector6f *directionalContribution,
	bool foundPoint, const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex,
	Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	computeNormalAndAngle<TVoxel, TIndex>(foundPoint, point, directionalContribution, voxelData, voxelIndex, lightSource, outNormal, angle);

	if (foundPoint) drawPixelNormal(outRendering, outNormal);
	else outRendering = Vector4u((uchar)0);
}

template<class TVoxel, class TIndex>
_CPU_AND_GPU_CODE_ inline void processPixelConfidence(DEVICEPTR(Vector4u) &outRendering, const CONSTPTR(Vector4f) & point,
	const Vector6f *directionalContribution,
	bool foundPoint, const CONSTPTR(TVoxel) *voxelData, const CONSTPTR(typename TIndex::IndexData) *voxelIndex, 
	Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	computeNormalAndAngle<TVoxel, TIndex>(foundPoint, TO_VECTOR3(point), directionalContribution, voxelData, voxelIndex, lightSource, outNormal, angle);

	if (foundPoint) drawPixelConfidence(outRendering, angle, point.w - 1.0f);
	else outRendering = Vector4u((uchar)0);
}

} // namespace ITMLib
