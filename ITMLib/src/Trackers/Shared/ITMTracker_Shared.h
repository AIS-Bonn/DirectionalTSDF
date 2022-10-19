//
// Created by Malte Splietker on 02.06.21.
//
// Loss functions and weights to use with normal least squares solver. From
// Babin 2019, Analysis of Robust Functions for Registration Algorithms
//

#pragma once

#include <Utils/ITMPixelUtils.h>
#include <ITMLib/Utils/ITMProjectionUtils.h>

namespace ITMLib
{

/**
 * Compute depth point (unproject depth value at x, y) and find corresponsing intensity value (project depth point into RGB image)
 *
 * @param x
 * @param y
 * @param out_points
 * @param out_rgb
 * @param in_rgb
 * @param in_depths
 * @param imageSize_rgb
 * @param imageSize_depth
 * @param intrinsics_rgb
 * @param intrinsics_depth
 * @param T_depthToRGB
 */
_CPU_AND_GPU_CODE_ inline void computeDepthPointAndColour(
	int x,
	int y,
	Vector4f* out_points,
	float* out_rgb,
	const float* in_rgb,
	const float* in_depths,
	const Vector2i& imageSize_rgb,
	const Vector2i& imageSize_depth,
	const Vector4f& intrinsics_rgb,
	const Vector4f& intrinsics_depth,
	const Matrix4f& T_depthToRGB)
{
	if (x >= imageSize_depth.x || y >= imageSize_depth.y) return;

	int sceneIdx = y * imageSize_depth.x + x;

	out_rgb[sceneIdx] = -1.f; // Mark as invalid
	out_points[sceneIdx] = Vector4f(0.f, 0.f, 0.f, -1.f); // Mark as invalid

	float depth_camera = in_depths[sceneIdx];

	// Invalid point
	if (depth_camera <= 0.f) return;

	const Vector3f pt_camera = unproject(x, y, depth_camera, intrinsics_depth);

	// Transform the point in the RGB sensor frame
	const Vector3f pt_image = T_depthToRGB * pt_camera;

	// Point behind the camera
	if (pt_image.z <= 0.f) return;

	// Project the point onto the RGB image plane
	const Vector2f pt_image_proj = project(pt_image, intrinsics_rgb);

	if (pt_image_proj.x < 0 || pt_image_proj.x >= imageSize_rgb.x - 1 ||
	    pt_image_proj.y < 0 || pt_image_proj.y >= imageSize_rgb.y - 1)
		return;

	out_rgb[sceneIdx] = interpolateBilinear_single(in_rgb, pt_image_proj, imageSize_rgb);
	out_points[sceneIdx] = Vector4f(pt_camera, 1.f);
}

/**
 * Huber-like? loss
 * Used by ExtendedTracker
 */
_CPU_AND_GPU_CODE_ inline float rho(float r, float huber_b)
{
	float tmp = std::abs(r) - huber_b;
	tmp = MAX(tmp, 0.0f);
	return r * r - tmp * tmp;
}

/**
 * First derivative of Huber loss
 * Used by ExtendedTracker
 */
_CPU_AND_GPU_CODE_ inline float rho_deriv(float r, float huber_b)
{
	return 2.0f * CLAMP(r, -huber_b, huber_b);
}

/**
 * Second derivative of Huber loss
 * Used by ExtendedTracker
 */
_CPU_AND_GPU_CODE_ inline float rho_deriv2(float r, float huber_b)
{
	return std::abs(r) < huber_b ? 2.0f : 0.0f;
}

class LossFunction
{
public:
	_CPU_AND_GPU_CODE_ virtual float Loss(float e) = 0;

	_CPU_AND_GPU_CODE_ virtual float Weight(float e) = 0;
};


class L2Loss : public LossFunction
{
public:
	_CPU_AND_GPU_CODE_ inline float Loss(float e) override
	{
		return e * e * 0.5;
	}

	_CPU_AND_GPU_CODE_ inline float Weight(float e) override
	{
		return 1;
	}
};

class HuberLoss : public LossFunction
{
public:
	float k = 0;

	_CPU_AND_GPU_CODE_ explicit HuberLoss(float k) : k(k)
	{}

	_CPU_AND_GPU_CODE_ inline float Loss(float e) override
	{
		float r_abs = std::abs(e);
		if (r_abs < k)
			return 0.5f * e * e;
		return k * (r_abs - 0.5f * k);
	}

	_CPU_AND_GPU_CODE_ inline float Weight(float e) override
	{
		float r_abs = std::abs(e);
		if (r_abs < k)
			return 1;
		return k / r_abs;
	}
};

class TukeyLoss : public LossFunction
{
public:
	float k = 0;

	_CPU_AND_GPU_CODE_ explicit TukeyLoss(float k) : k(k)
	{}

	_CPU_AND_GPU_CODE_ inline float Loss(float e) override
	{
		if (std::abs(e) < k)
			return k * k * (1 - powf(1 - (e / k) * (e / k), 3)) / 2;
		return k * k / 2;
	}

	_CPU_AND_GPU_CODE_ inline float Weight(float e) override
	{
		if (std::abs(e) < k)
		{
			float a = (1 - (e / k) * (e / k));
			return a * a;
		}
		return 0;
	}
};

class CauchyLoss : public LossFunction
{
public:
	float k = 0;

	_CPU_AND_GPU_CODE_ explicit CauchyLoss(float k) : k(k)
	{}

	_CPU_AND_GPU_CODE_ inline float Loss(float e) override
	{
		return 0.5f * k * k * logf(1 + (e / k) * (e / k));
	}

	_CPU_AND_GPU_CODE_ inline float Weight(float e) override
	{
		return 1 / (1 + (e / k) * (e / k));
	}
};

} // ITMLib