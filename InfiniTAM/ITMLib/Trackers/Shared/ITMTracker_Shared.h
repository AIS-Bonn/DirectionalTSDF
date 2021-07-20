//
// Created by Malte Splietker on 02.06.21.
//

#pragma once

#include <ITMLib/Utils/ITMPixelUtils.h>
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
	const Vector2i &imageSize_rgb,
	const Vector2i &imageSize_depth,
	const Vector4f &intrinsics_rgb,
	const Vector4f &intrinsics_depth,
	const Matrix4f &T_depthToRGB)
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

} // ITMLib