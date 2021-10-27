// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <math_constants.h>
#include <ITMLib/Trackers/Shared/ITMICPTracker_Shared.h>
#include <ITMLib/Engines/Reconstruction/Shared/ITMFusionWeight.hpp>
#include "ITMLib/Objects/Scene/ITMDirectional.h"
#include "ITMLib/Engines/Reconstruction/Interface/ITMSceneReconstructionEngine.h"
#include "ITMLib/Utils/ITMProjectionUtils.h"
#include "ITMLib/Utils/ITMFilter.h"

namespace ITMLib
{

static const int MAX_RENDERING_BLOCKS = 65536 * 4;
//static const int MAX_RENDERING_BLOCKS = 16384;
static const int minmaximg_subsample = 8;

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

static const int renderingBlockSizeX = 16;
static const int renderingBlockSizeY = 16;

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
_CPU_AND_GPU_CODE_ inline bool ProjectSingleBlock(const Vector3s& blockPos, const Matrix4f& pose,
                                                  const Vector4f& intrinsics,
                                                  const Vector2i& imgSize, float voxelSize,
                                                  Vector2i& upperLeft, Vector2i& lowerRight,
                                                  Vector2f& zRange)
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

_CPU_AND_GPU_CODE_ inline void CreateRenderingBlocks(RenderingBlock* renderingBlockList, int offset,
                                                     const Vector2i& upperLeft,
                                                     const Vector2i& lowerRight,
                                                     const Vector2f& zRange)
{
	// split bounding box into 16x16 pixel rendering blocks
	for (int by = 0; by < ceil((float) (1 + lowerRight.y - upperLeft.y) / renderingBlockSizeY); ++by)
	{
		for (int bx = 0; bx < ceil((float) (1 + lowerRight.x - upperLeft.x) / renderingBlockSizeX); ++bx)
		{
			if (offset >= MAX_RENDERING_BLOCKS) return;
			//for each rendering block: add it to the list
			RenderingBlock& b(renderingBlockList[offset++]);
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
forwardProjectPixel(Vector4f pixel, const Matrix4f& M, const Vector4f projParams,
                    const Vector2i& imgSize)
{
	pixel.w = 1;
	pixel = M * pixel;

	Vector2f pt_image;
	pt_image.x = projParams.x * pixel.x / pixel.z + projParams.z;
	pt_image.y = projParams.y * pixel.y / pixel.z + projParams.w;

	if ((pt_image.x < 0) || (pt_image.x > imgSize.x - 1) || (pt_image.y < 0) || (pt_image.y > imgSize.y - 1)) return -1;

	return (int) (pt_image.x + 0.5f) + (int) (pt_image.y + 0.5f) * imgSize.x;
}

template<class TIndex, class TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline void computeNormalAndAngleTSDF(bool& foundPoint, const Vector3f& point,
                                                         const Map<TIndex, TVoxel*, Args...>& tsdf,
                                                         const float oneOverVoxelSize,
                                                         const Vector3f& lightSource,
                                                         Vector3f& outNormal, float& angle)
{
	if (!foundPoint) return;

	outNormal = computeSingleNormalFromSDF(tsdf, point * oneOverVoxelSize).normalised();

	Vector3f lightDirection = (point - lightSource).normalised();
	angle = dot(outNormal, -lightDirection);
	if (angle <= 0.0) foundPoint = false;
}

template<bool useSmoothing>
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

	Vector3f lightDirection = (pointsRay[x + y * imgSize.x].toVector3() - lightSource).normalised();
	angle = dot(outNormal, -lightDirection);
	if (angle <= 0.0) foundPoint = false;
}

template<bool useSmoothing>
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
}

_CPU_AND_GPU_CODE_ inline void drawPixelGrey(Vector4u& dest, const float& angle)
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
	switch (sector)
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
drawPixelConfidence(Vector4u& dest, const float& angle,
                    const float& normalizedConfidence)
{
	float confidenceNorm = CLAMP(normalizedConfidence, 0, 1.0f);

	Vector4f color;
	color = Vector4f(HSVtoRGB(confidenceNorm * 120, 1, 1) * 255, 255);

	Vector4f outRes = (0.8f * angle + 0.2f) * color;
	dest = TO_UCHAR4(outRes);
}

_CPU_AND_GPU_CODE_ inline void drawPixelNormal(Vector4u& dest, const Vector3f& normal_obj)
{
	dest.r = (uchar) ((0.3f + (-normal_obj.r + 1.0f) * 0.35f) * 255.0f);
	dest.g = (uchar) ((0.3f + (-normal_obj.g + 1.0f) * 0.35f) * 255.0f);
	dest.b = (uchar) ((0.3f + (-normal_obj.b + 1.0f) * 0.35f) * 255.0f);
}

template<typename TIndex, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline bool castRayDefaultTSDF(Vector4f& pt_out,
                                                  float& distance_out,
                                                  int x, int y, const Map<TIndex, ITMVoxel*, Args...>& tsdf,
                                                  Matrix4f invM, Vector4f invProjParams,
                                                  const ITMSceneParams& sceneParams,
                                                  const Vector2f& minMaxImg,
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

		TIndex index;
		unsigned short linearIdx;
		voxelIdxToIndexAndOffset(index, linearIdx,
		                         Vector3i((int) ROUND(pt_result.x), (int) ROUND(pt_result.y), (int) ROUND(pt_result.z)),
		                         direction);

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

		sdfValue = readWithConfidenceFromSDF_float_interpolated(found, confidence, tsdf, pt_result, sceneParams.maxW,
		                                                        direction);

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

template<class TIndex, class TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline bool castRay(Vector4f& pt_out,
                                       int x, int y, const Map<TIndex, TVoxel*, Args...>& tsdf,
                                       Matrix4f invM, Vector4f invProjParams,
                                       const ITMSceneParams& sceneParams,
                                       const Vector2f& minMaxImg)
{
	float distance;
	return castRayDefaultTSDF(pt_out, distance, x, y, tsdf, invM,
	                          invProjParams, sceneParams, minMaxImg);
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

template<bool useSmoothing>
_CPU_AND_GPU_CODE_ inline void processPixelICP(Vector4f* pointsMap, Vector4f* normalsMap,
                                               const Vector4f* pointsRay, const Vector4f* normalsRay,
                                               const Vector2i& imgSize,
                                               const int& x, const int& y, float voxelSize,
                                               const Vector3f& lightSource)
{
	Vector3f outNormal;
	float angle;

	int locId = x + y * imgSize.x;
	Vector4f point = pointsRay[locId];

	bool foundPoint = point.w > 0.0f;

	computeNormalAndAngle<useSmoothing>(pointsRay, normalsRay, lightSource, imgSize, x, y, foundPoint, outNormal, angle);

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

template<class TIndex, class TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline void processPixelDepthShaded_SDFNormals(Vector4u& outRendering, const Vector4f& point,
                                                                  const Map<TIndex, TVoxel*, Args...>& tsdf,
                                                                  const float oneOverVoxelSize,
                                                                  const Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	bool foundPoint = point.w > 0;
	computeNormalAndAngleTSDF(foundPoint, point.toVector3(), tsdf, oneOverVoxelSize, lightSource, outNormal, angle);

	if (foundPoint) drawPixelGrey(outRendering, angle);
	else outRendering = Vector4u((uchar) 0);
}


template<bool useSmoothing>
_CPU_AND_GPU_CODE_ inline void
processPixelDepthShaded_ImageNormals(Vector4u* outRendering,
                                     const Vector4f* pointsRay, const Vector4f* normalsRay,
                                     const Vector2i& imgSize, const int x, const int y,
                                     const Vector3f& lightSource)
{
	Vector3f outNormal;
	float angle;

	int locId = x + y * imgSize.x;
	Vector4f point = pointsRay[locId];

	bool foundPoint = point.w > 0.0f;
	computeNormalAndAngle<useSmoothing>(pointsRay, normalsRay, lightSource, imgSize,
	                                    x, y, foundPoint, outNormal, angle);

	if (foundPoint) drawPixelGrey(outRendering[locId], angle);
	else outRendering[locId] = Vector4u((uchar) 0);
}

template<bool useSmoothing>
_CPU_AND_GPU_CODE_ inline void
processPixelNormals_ImageNormals(Vector4u* outRendering, const Vector4f* pointsRay, const Vector4f* normalsRay,
                                 const Vector2i& imgSize, const int& x, const int& y,
                                 float voxelSize, Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	int locId = x + y * imgSize.x;
	Vector4f point = pointsRay[locId];

	bool foundPoint = point.w > 0.0f;
	computeNormalAndAngle<useSmoothing>(pointsRay, normalsRay, lightSource, imgSize,
	                                    x, y, foundPoint, outNormal, angle);

	if (foundPoint) drawPixelNormal(outRendering[locId], outNormal);
	else outRendering[locId] = Vector4u((uchar) 0);
}

template<bool useSmoothing>
_CPU_AND_GPU_CODE_ inline void
processPixelConfidence_ImageNormals(Vector4u* outRendering, const Vector4f* pointsRay,
                                    const Vector4f* normalsRay,
                                    const Vector2i& imgSize, const int& x,
                                    const int& y, const ITMSceneParams& sceneParams, Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	int locId = x + y * imgSize.x;
	Vector4f point = pointsRay[locId];

	bool foundPoint = point.w > 0.0f;
	computeNormalAndAngle<useSmoothing>(pointsRay, normalsRay, lightSource, imgSize,
	                                    x, y, foundPoint, outNormal, angle);

	if (foundPoint)
		drawPixelConfidence(outRendering[locId], angle,
		                    point.w / static_cast<float>(sceneParams.maxW));
	else outRendering[locId] = Vector4u((uchar) 0);
}

template<class TVoxel>
_CPU_AND_GPU_CODE_ inline void processPixelDepthColour(
	Vector4u& outRendering, const Vector4f& point,
	const Matrix4f& T_CW, const float maxDepth)
{
	bool foundPoint = point.w > 0;
	Vector4f pointCamera = T_CW * Vector4f(point.toVector3(), 1);
	float depth = pointCamera.z;

	if (not foundPoint or depth <= 0.0f)
	{
		outRendering = Vector4u(0, 0, 0, 255);
		return;
	}

	float hue = (1 - (maxDepth - depth) / maxDepth) * 240;

	outRendering = Vector4f(HSVtoRGB(hue, 1, 1) * 255, 255).toUChar();
}

template<class TIndex, class TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline void processPixelColour(
	Vector4u& outRendering, const Vector4f& point, const Map<TIndex, TVoxel*, Args...>& tsdf,
	const float oneOverVoxelSize, const Vector3f lightSource)
{
	bool foundPoint = point.w > 0;

//	float angle;
//	Vector3f outNormal;
//	computeNormalAndAngleTSDF<TVoxel, TIndex>(foundPoint, point.toVector3(), directionalContribution, voxelData, voxelIndex,
//	                                      lightSource, outNormal, angle);

	if (foundPoint)
	{
		Vector4f clr = readFromSDF_color4u_interpolated(tsdf, point.toVector3() * oneOverVoxelSize);
		outRendering.x = (uchar) (clr.x * 255.0f);
		outRendering.y = (uchar) (clr.y * 255.0f);
		outRendering.z = (uchar) (clr.z * 255.0f);
		outRendering.w = 255;

	} else outRendering = Vector4u((uchar) 0);
}

template<class TIndex, class TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline void
processPixelNormal_SDFNormals(Vector4u& outRendering, const Vector4f& point,
                              const Map<TIndex, TVoxel*, Args...>& tsdf,
                              const float oneOverVoxelSize,
                              const Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	bool foundPoint = point.w > 0;
	computeNormalAndAngleTSDF(foundPoint, point.toVector3(), tsdf, oneOverVoxelSize, lightSource, outNormal, angle);

	if (foundPoint) drawPixelNormal(outRendering, outNormal);
	else outRendering = Vector4u((uchar) 0);
}

template<class TIndex, class TVoxel, template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ inline void
processPixelConfidence_SDFNormals(Vector4u& outRendering, const Vector4f& point,
                                  const Map<TIndex, TVoxel*, Args...>& tsdf,
                                  const ITMSceneParams& sceneParams,
                                  const Vector3f lightSource)
{
	Vector3f outNormal;
	float angle;

	bool foundPoint = point.w > 0;
	computeNormalAndAngleTSDF(foundPoint, point.toVector3(), tsdf, sceneParams.oneOverVoxelSize, lightSource, outNormal,
	                          angle);

	if (foundPoint)
		drawPixelConfidence(outRendering, angle, point.w / static_cast<float>(sceneParams.maxW));
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
//	float angle = -(sceneRenderingPose * normalsRay[locId]).z;

	if (!isValidPoint)
	{// or angle <= 0.0) {
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

/**
 * compute combined voxel parameters (sdf, weight, colour, colour_weight) at voxelPos
 * @tparam Map
 * @tparam Args
 * @param voxelPos
 * @param tsdf
 * @param invM
 * @param voxelSize
 * @param mu
 * @param maxW
 * @return
 */
template<template<typename, typename...> class Map, typename... Args>
_CPU_AND_GPU_CODE_ ITMVoxel
combineDirectionalTSDFViewPoint(const Vector3f voxelPos,
                                const Map<ITMIndexDirectional, ITMVoxel*, Args...>& tsdf,
                                const Matrix4f invM, const float voxelSize, const float mu, const int maxW)
{

	Vector3f rayDirection = (voxelPos * voxelSize - (invM * Vector4f(0, 0, 0, 1)).toVector3()).normalised();

	Vector3f colorCombined(0, 0, 0);
	float sdfCombined = 0;
	float weightCombined = 0;
	float colorWeightCombined = 0;

	Vector3f colorCombinedNoGradient(0, 0, 0);
	Vector3f gradientCombined(0, 0, 0);
	float sdfCombinedNoGradient = 0;
	float weightCombinedNoGradient = 0;
	float colorWeightCombinedNoGradient = 0;

	Vector3f colorFreeSpace(0, 0, 0);
	Vector3f gradientFreeSpace(0, 0, 0);
	float freeSpaceWeight = 0;
	float freeSpaceSDF = 0;

	for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
	{
		float confidence = 0;
		bool found = false;
		float sdf = readWithConfidenceFromSDF_float_uninterpolated(found, confidence, tsdf, voxelPos,
		                                                           maxW, TSDFDirection(directionIdx));

		if (not found)
			continue;

		Vector3f color;
		float color_w = 1;
		if (RENDER_DIRECTION_COLORS == 1)
			color = TSDFDirectionColor[directionIdx];
		else
			color = readFromSDF_color4u_uninterpolated(color_w, tsdf, voxelPos, maxW,
			                                           TSDFDirection(directionIdx)).toVector3();
		Vector3f gradient = computeSingleNormalFromSDF(tsdf, voxelPos.toIntFloor(), TSDFDirection(directionIdx),
		                                               voxelSize / mu);

		float weight =
			DirectionWeight(DirectionAngle(gradient, TSDFDirection(directionIdx)))
			* ORUtils::dot(gradient, -rayDirection)
			* confidence;
		weight = MAX(weight, 0);

		float weightNoGradient = MAX(0,
		                             confidence
		                             * ORUtils::dot(-rayDirection, TSDFDirectionVector[directionIdx])
//		                             * DirectionWeight(DirectionAngle(-rayDirection, TSDFDirection(directionIdx)))
		);
		weightCombinedNoGradient += weightNoGradient;
		sdfCombinedNoGradient += weightNoGradient * sdf;
		colorWeightCombinedNoGradient += weightNoGradient * color_w;
		colorCombinedNoGradient += weightNoGradient * color_w * color;

		sdfCombined += weight * sdf;
		weightCombined += weight;
		colorWeightCombined += weight * color_w;
		colorCombined += weight * color_w * color;
		gradientCombined += weight * gradient;
	}

	bool hasGradient = gradientCombined.x != 0 or gradientCombined.y != 0 or gradientCombined.z != 0;
	bool hasFreeSpaceGradient = gradientFreeSpace.x != 0 or gradientFreeSpace.y != 0 or gradientFreeSpace.z != 0;

	if (weightCombined > 0)
	{
		sdfCombined /= weightCombined;
		if (hasGradient) gradientCombined = gradientCombined.normalised();
	} else if (weightCombinedNoGradient > 0)
	{
		sdfCombined = sdfCombinedNoGradient / weightCombinedNoGradient;
		weightCombined = weightCombinedNoGradient;
		gradientCombined = Vector3f(0, 0, 0);
	} else
	{
		sdfCombined = 1;
		weightCombined = 0;
		gradientCombined = Vector3f(0, 0, 0);
	}

	if (colorWeightCombined > 0)
	{
		colorCombined /= colorWeightCombined;
	} else if (colorWeightCombinedNoGradient > 0)
	{
		colorCombined = colorCombinedNoGradient / colorWeightCombinedNoGradient;
		colorWeightCombined = colorWeightCombinedNoGradient;
	} else
	{
		colorCombined = Vector3f(0, 0, 0);
		colorWeightCombined = 0;
	}

	if (freeSpaceWeight > 0)
	{
		freeSpaceSDF /= freeSpaceWeight;
		colorFreeSpace /= freeSpaceWeight;
		if (hasFreeSpaceGradient)
			gradientFreeSpace = gradientFreeSpace.normalised();
	}

	ITMVoxel combinedVoxel;
	if (freeSpaceWeight > 0
	    and ((hasGradient and hasFreeSpaceGradient
	          and dot(gradientFreeSpace, gradientCombined) <
	              0.707 // if same surface, use normal combination instead of free space (prevent dents in surface)
	          and dot(gradientFreeSpace, gradientCombined) > -0.707 // if opposite surface, don't carve
	         ) or (not hasGradient and not hasFreeSpaceGradient)))
	{
		combinedVoxel.sdf = ITMVoxel::floatToValue(freeSpaceSDF);
		combinedVoxel.w_depth = ITMVoxel::floatToWeight(freeSpaceWeight, maxW);
		combinedVoxel.clr = TO_UCHAR3(colorFreeSpace * 255.0f);
		combinedVoxel.w_color = ITMVoxel::floatToWeight(freeSpaceWeight, maxW);
	} else
	{
		combinedVoxel.sdf = ITMVoxel::floatToValue(sdfCombined);
		combinedVoxel.w_depth = ITMVoxel::floatToWeight(weightCombined, maxW);
		combinedVoxel.clr = TO_UCHAR3(colorCombined * 255.0f);
		combinedVoxel.w_color = ITMVoxel::floatToWeight(colorWeightCombined, maxW);
	}
	return combinedVoxel;
}

} // namespace ITMLib
