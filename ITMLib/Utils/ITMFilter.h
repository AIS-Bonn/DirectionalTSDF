//
// Created by Malte Splietker on 08.06.20.
//

#pragma once

namespace ITMLib
{

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

/** Performs bilateral filtering for a single normal pixel
 *
 * tuning guide:
 * sigma_d: larger values -> higher weight for pixels further away
 * sigma_r: smaller values -> lower weight for normals, that vary from center normal
 *
 * @param sigma_d parameter for spatial filter
 * @param sigma_r parameter for gradient filter (weighting difference in distance)
 * @param radius pixel radius around (x,y)
 */
_CPU_AND_GPU_CODE_ inline Vector4f
computeNormalBilateralFiltered(const Vector4f* normals, const float sigma_d, const float sigma_r, const int radius,
                               const int x, const int y, const Vector2i& imgSize)
{
	Vector4f result(0, 0, 0, -1);

	const Vector4f center = normals[y * imgSize.x + x];
	if (center.w < 0)
		return result;

	Vector3f sum(0, 0, 0);
	float sum_weight = 0.0f;

	float sigma_d_factor = 1 / (2.0f * sigma_d * sigma_d);
	float sigma_r_factor = 1 / (2.0f * sigma_r * sigma_r);

	for (int i = x - radius; i <= x + radius; i++)
	{
		for (int j = y - radius; j <= y + radius; j++)
		{
			if (i < 0 or j < 0 or i >= imgSize.x or j >= imgSize.y)
				continue;

			const Vector4f value = normals[j * imgSize.x + i];

			if (value.w == -1)
				continue;

			const float weight =
				gaussD(sigma_d_factor, i - x, j - y) * gaussR(sigma_r_factor, length((value - center).toVector3()));

			sum += weight * value.toVector3();
			sum_weight += weight;
		}
	}

	if (sum_weight >= 0.0f)
	{
		result = Vector4f((sum / sum_weight).normalised(), 1.0f);
	}

	return result;
}

/** Performs bilateral filtering for a depth pixel
 *
 * tuning guide:
 * sigma_d: larger values -> higher weight for pixels further away (more bluring)
 * sigma_r: smaller values -> lower weight for values, that vary from center depth (more edge-awareness)
 *
 * @param sigma_d parameter for spatial filter
 * @param sigma_r parameter for gradient filter (weighting difference in distance)
 * @param radius pixel radius around (x,y)
 */
_CPU_AND_GPU_CODE_ inline float
computeDepthBilateralFiltered(const float* depth, const float sigma_d, const float sigma_r, const int radius,
                              const int x, const int y, const Vector2i& imgSize)
{
	const float center = depth[y * imgSize.x + x];
	if (center < 0)
		return center;

	float sum = 0;
	float sum_weight = 0.0f;

	float sigma_d_factor = 1 / (2.0f * sigma_d * sigma_d);
//	float sigma_r_factor = 1 / (2.0f * sigma_r * sigma_r);
	float sigma_r_factor = 1 / (2.0f * sigma_r * sigma_r * center *
	                            center); // depth-dependent gradient filter (further away->higher noise->higher sigma_r)

	for (int i = x - radius; i <= x + radius; i++)
	{
		for (int j = y - radius; j <= y + radius; j++)
		{
			if (i < 0 or j < 0 or i >= imgSize.x or j >= imgSize.y)
				continue;

			const float value = depth[j * imgSize.x + i];

			if (value < 0)
				continue;

			const float weight = gaussD(sigma_d_factor, i - x, j - y) * gaussR(sigma_r_factor, fabs(value - center));

			sum += weight * value;
			sum_weight += weight;
		}
	}

	if (sum_weight >= 0.0f)
		return sum / sum_weight;

	return -1;
}

} // ITMLib
