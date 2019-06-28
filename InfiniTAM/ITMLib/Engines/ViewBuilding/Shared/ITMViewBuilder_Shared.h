// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMLib/Utils/ITMMath.h"
#include "ITMLib/Utils/ITMProjectionUtils.h"
#include "ORUtils/PlatformIndependence.h"

namespace ITMLib
{

_CPU_AND_GPU_CODE_ inline void convertDisparityToDepth(DEVICEPTR(float) *d_out, int x, int y, const CONSTPTR(short) *d_in,
	Vector2f disparityCalibParams, float fx_depth, Vector2i imgSize)
{
	int locId = x + y * imgSize.x;

	short disparity = d_in[locId];
	float disparity_tmp = disparityCalibParams.x - (float)(disparity);
	float depth;

	if (disparity_tmp == 0) depth = 0.0;
	else depth = 8.0f * disparityCalibParams.y * fx_depth / disparity_tmp;

	d_out[locId] = (depth > 0) ? depth : -1.0f;
}

#ifndef __METALC__

_CPU_AND_GPU_CODE_ inline void convertDepthAffineToFloat(DEVICEPTR(float) *d_out, int x, int y, const CONSTPTR(short) *d_in, Vector2i imgSize, Vector2f depthCalibParams)
{
	int locId = x + y * imgSize.x;

	short depth_in = d_in[locId];
	d_out[locId] = ((depth_in <= 0)||(depth_in > 32000)) ? -1.0f : (float)depth_in * depthCalibParams.x + depthCalibParams.y;
}

#define MEAN_SIGMA_L 1.2232f
_CPU_AND_GPU_CODE_ inline void filterDepth(DEVICEPTR(float) *imageData_out, const CONSTPTR(float) *imageData_in, int x, int y, Vector2i imgDims)
{
	float z, tmpz, dz, final_depth = 0.0f, w, w_sum = 0.0f;

	z = imageData_in[x + y * imgDims.x];
	if (z < 0.0f) { imageData_out[x + y * imgDims.x] = -1.0f; return; }

	float sigma_z = 1.0f / (0.0012f + 0.0019f*(z - 0.4f)*(z - 0.4f) + 0.0001f / sqrt(z) * 0.25f);

	for (int i = -2, count = 0; i <= 2; i++) for (int j = -2; j <= 2; j++, count++)
	{
		tmpz = imageData_in[(x + j) + (y + i) * imgDims.x];
		if (tmpz < 0.0f) continue;
		dz = (tmpz - z); dz *= dz;
		w = exp(-0.5f * ((abs(i) + abs(j))*MEAN_SIGMA_L*MEAN_SIGMA_L + dz * sigma_z * sigma_z));
		w_sum += w;
		final_depth += w*tmpz;
	}

	final_depth /= w_sum;
	imageData_out[x + y*imgDims.x] = final_depth;
}

/**
 * Implementation from BundleFusion
 * Copyright (c) 2017 by Angela Dai and Matthias Niessner
 */
_CPU_AND_GPU_CODE_
inline float gaussD(float factor, int x, int y)
{
	return exp(-((x * x + y * y) * factor));
}

/**
 * Implementation from BundleFusion
 * Copyright (c) 2017 by Angela Dai and Matthias Niessner
 */
_CPU_AND_GPU_CODE_
inline float gaussR(float factor, float dist)
{
	return exp(-(dist * dist) * factor);
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
_CPU_AND_GPU_CODE_ inline void filterNormals(Vector4f *normals_out, const Vector4f *normals_in,
	float sigma_d, float sigma_r, int x, int y, Vector2i imgDims)
{
	const int width = imgDims.x;
	const int height = imgDims.y;

	if (x >= width or y >= height)
		return;

	const uint idx = y * width + x;

	normals_out[idx] = Vector4f(0, 0, 0, -1);

	const Vector4f center = normals_in[idx];
	if (center.w == -1)
		return;

	Vector3f sum(0, 0, 0);
	float sum_weight = 0.0f;

	float sigma_d_factor = 1 / (2.0f * sigma_d * sigma_d);
	float sigma_r_factor = 1 / (2.0f * sigma_r * sigma_r);

	const uchar radius = static_cast<uchar>(sigma_d);
	for (int i = x - radius; i <= x + radius; i++)
	{
		for (int j = y - radius; j <= y + radius; j++)
		{
			if (i < 0 or j < 0 or i >= width or j >= height)
				continue;

			const Vector4f value = normals_in[j * width + i];

			if (value.w == -1)
				continue;

			const float weight = gaussD(sigma_d_factor, i - x, j - y) * gaussR(sigma_r_factor, length(value - center));

			sum += weight * value.toVector3();
			sum_weight += weight;
		}
	}

	if (sum_weight >= 0.0f)
	{
		normals_out[idx] = Vector4f((sum / sum_weight).normalised(), 1.0f);
	}
}


_CPU_AND_GPU_CODE_ inline void computeNormalAndWeight(const CONSTPTR(float) *depth_in, DEVICEPTR(Vector4f) *normal_out, DEVICEPTR(float) *sigmaZ_out, int x, int y, Vector2i imgDims, Vector4f intrinparam)
{
	Vector3f outNormal;

	int idx = x + y * imgDims.x;

	float z = depth_in[x + y * imgDims.x];
	if (z < 0.0f)
	{
		normal_out[idx].w = -1.0f;
		sigmaZ_out[idx] = -1;
		return;
	}

	// first compute the normal
	Vector3f diff_x(0.0f, 0.0f, 0.0f), diff_y(0.0f, 0.0f, 0.0f);

	Vector4f invProjParams_d = invertProjectionParams(intrinparam);
	Vector3f xp1_y = reprojectImagePoint(x + 1, y, depth_in[(x + 1) + y * imgDims.x], invProjParams_d);
	Vector3f xm1_y = reprojectImagePoint(x - 1, y, depth_in[(x - 1) + y * imgDims.x], invProjParams_d);
	Vector3f x_yp1 = reprojectImagePoint(x, y + 1, depth_in[x + (y + 1) * imgDims.x], invProjParams_d);
	Vector3f x_ym1 = reprojectImagePoint(x, y - 1, depth_in[x + (y - 1) * imgDims.x], invProjParams_d);

	if (xp1_y.z <= 0 or x_yp1.z <= 0 or xm1_y.z <= 0 or x_ym1.z <= 0 or abs(xp1_y.z - z) > 0.02 or abs(xm1_y.z - z) > 0.02)
	{
		normal_out[idx].w = -1.0f;
		sigmaZ_out[idx] = -1;
		return;
	}

	// gradients x and y
	diff_x = xp1_y - xm1_y, diff_y = x_yp1 - x_ym1;

	outNormal = cross(diff_y, diff_x);

	if (outNormal.x == 0.0f && outNormal.y == 0 && outNormal.z == 0)
	{
		normal_out[idx].w = -1.0f;
		sigmaZ_out[idx] = -1;
		return;
	}
	outNormal = outNormal.normalised();

	normal_out[idx] = Vector4f(outNormal, 1.0f);

	// now compute weight
	float theta = acos(outNormal.z);
	float theta_diff = theta / (PI*0.5f - theta);

	sigmaZ_out[idx] = (0.0012f + 0.0019f * (z - 0.4f) * (z - 0.4f) + 0.0001f / sqrt(z) * theta_diff * theta_diff);
}

} // namespace ITMLib

#endif
