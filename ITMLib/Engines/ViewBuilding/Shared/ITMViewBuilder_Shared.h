// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMLib/Utils/ITMFilter.h"
#include "ITMLib/Utils/ITMMath.h"
#include "ITMLib/Utils/ITMProjectionUtils.h"
#include "ORUtils/PlatformIndependence.h"

namespace ITMLib
{

_CPU_AND_GPU_CODE_ inline void convertDisparityToDepth(float* d_out, int x, int y, const short* d_in,
                                                       Vector2f disparityCalibParams, float fx_depth, Vector2i imgSize)
{
	int locId = x + y * imgSize.x;

	short disparity = d_in[locId];
	float disparity_tmp = disparityCalibParams.x - (float) (disparity);
	float depth;

	if (disparity_tmp == 0) depth = 0.0;
	else depth = 8.0f * disparityCalibParams.y * fx_depth / disparity_tmp;

	d_out[locId] = (depth > 0) ? depth : -1.0f;
}

_CPU_AND_GPU_CODE_ inline void
convertDepthAffineToFloat(float* d_out, int x, int y, const short* d_in, Vector2i imgSize, Vector2f depthCalibParams)
{
	int locId = x + y * imgSize.x;

	short depth_in = d_in[locId];
	d_out[locId] = ((depth_in <= 0) || (depth_in > 32000)) ? -1.0f : (float) depth_in * depthCalibParams.x +
	                                                                 depthCalibParams.y;
}

_CPU_AND_GPU_CODE_ inline void filterDepth(float* imageData_out, const float* imageData_in,
                                           const float sigma_d, const float sigma_r, int x, int y, Vector2i imgDims)
{
	if (x >= imgDims.x or y >= imgDims.y)
		return;

	imageData_out[y * imgDims.x + x] = computeDepthBilateralFiltered(imageData_in, sigma_d, sigma_r, 5, x, y, imgDims);
}

/**
 * Bilateral filtering for normals
 *
 * Implementation from BundleFusion
 * Copyright (c) 2017 by Angela Dai and Matthias Niessner
 *
 * @param normals_out
 * @param normals_in
 * @param sigma_d
 * @param sigma_r
 * @param x
 * @param y
 * @param imgDims
 */
_CPU_AND_GPU_CODE_ inline void filterNormals(Vector4f* normals_out, const Vector4f* normals_in,
                                             float sigma_d, float sigma_r, int x, int y, Vector2i imgDims)
{
	if (x >= imgDims.x or y >= imgDims.y)
		return;

	normals_out[y * imgDims.x + x] = computeNormalBilateralFiltered(normals_in, sigma_d, sigma_r, 5, x, y, imgDims);
}


_CPU_AND_GPU_CODE_ inline void
computeNormalAndWeight(const float* depth_in, Vector4f* normal_out, int x, int y, Vector2i imgDims,
                       Vector4f intrinparam)
{
	Vector3f outNormal;

	int idx = x + y * imgDims.x;

	float z = depth_in[x + y * imgDims.x];
	if (z < 0.0f)
	{
		normal_out[idx] = Vector4f(0, 0, 0, -1.0f);
		return;
	}

	Vector4f invProjParams_d = invertProjectionParams(intrinparam);

	Vector4f x_y = Vector4f(reprojectImagePoint(x, y, depth_in[(x) + y * imgDims.x], invProjParams_d),
	                        depth_in[(x) + y * imgDims.x] > 0 ? 1 : 0);
	Vector4f xp_y = Vector4f(reprojectImagePoint(x + 1, y, depth_in[(x + 1) + y * imgDims.x], invProjParams_d),
	                         depth_in[(x + 1) + y * imgDims.x] > 0 ? 1 : 0);
	Vector4f xm_y = Vector4f(reprojectImagePoint(x - 1, y, depth_in[(x - 1) + y * imgDims.x], invProjParams_d),
	                         depth_in[(x - 1) + y * imgDims.x] > 0 ? 1 : 0);
	Vector4f x_yp = Vector4f(reprojectImagePoint(x, y + 1, depth_in[x + (y + 1) * imgDims.x], invProjParams_d),
	                         depth_in[x + (y + 1) * imgDims.x] > 0 ? 1 : 0);
	Vector4f x_ym = Vector4f(reprojectImagePoint(x, y - 1, depth_in[x + (y - 1) * imgDims.x], invProjParams_d),
	                         depth_in[x + (y - 1) * imgDims.x] > 0 ? 1 : 0);

	Vector3f sum(0, 0, 0);
	float weightSum = 0;

	if (xp_y.w > 0 and x_yp.w > 0)
	{
		Vector3f diff_x = (xp_y - x_y).toVector3();
		Vector3f diff_y = (x_yp - x_y).toVector3();
		if (ORUtils::length(diff_x) < 3 and ORUtils::length(diff_y) < 3)
		{
			weightSum += 1;
			Vector3f normal = -ORUtils::cross(diff_x, diff_y).normalised();
			sum += normal;
		}
	}
	if (xm_y.w > 0 and x_yp.w > 0)
	{
		Vector3f diff_x = (x_y - xm_y).toVector3();
		Vector3f diff_y = (x_yp - x_y).toVector3();
		if (ORUtils::length(diff_x) < 3 and ORUtils::length(diff_y) < 3)
		{
			weightSum += 1;
			Vector3f normal = -ORUtils::cross(diff_x, diff_y).normalised();
			sum += normal;
		}
	}
	if (xm_y.w > 0 and x_ym.w > 0)
	{
		Vector3f diff_x = (x_y - xm_y).toVector3();
		Vector3f diff_y = (x_y - x_ym).toVector3();
		if (ORUtils::length(diff_x) < 3 and ORUtils::length(diff_y) < 3)
		{
			weightSum += 1;
			Vector3f normal = -ORUtils::cross(diff_x, diff_y).normalised();
			sum += normal;
		}
	}
	if (xp_y.w > 0 and x_ym.w > 0)
	{
		Vector3f diff_x = (xp_y - x_y).toVector3();
		Vector3f diff_y = (x_y - x_ym).toVector3();
		if (ORUtils::length(diff_x) < 3 and ORUtils::length(diff_y) < 3)
		{
			weightSum += 1;
			Vector3f normal = -ORUtils::cross(diff_x, diff_y).normalised();
			sum += normal;
		}
	}
	if (weightSum == 0)
	{
		normal_out[idx] = Vector4f(0, 0, 0, -1);
		return;
	}
	normal_out[idx] = Vector4f((sum / weightSum).normalised(), 1.0f);
	float theta = acos(outNormal.z);
}

} // namespace ITMLib
