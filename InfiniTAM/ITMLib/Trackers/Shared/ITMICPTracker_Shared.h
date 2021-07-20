// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../Utils/ITMPixelUtils.h"
#include "Utils/ITMProjectionUtils.h"

namespace ITMLib
{

template<bool shortIteration, bool rotationOnly>
_CPU_AND_GPU_CODE_ inline bool computePerPointError(THREADPTR(float) &error,
                                                         const THREADPTR(int) & x, const THREADPTR(int) & y,
                                                         const CONSTPTR(float) &depth, const CONSTPTR(Vector2i) & viewImageSize, const CONSTPTR(Vector4f) & viewIntrinsics, const CONSTPTR(Vector2i) & sceneImageSize,
                                                         const CONSTPTR(Vector4f) & sceneIntrinsics, const CONSTPTR(Matrix4f) & approxInvPose, const CONSTPTR(Matrix4f) & scenePose, const CONSTPTR(Vector4f) *pointsMap)
{
	if (depth <= 1e-8f) return false; //check if valid -- != 0.0f

	Vector4f tmp3Dpoint, tmp3Dpoint_reproj; Vector3f ptDiff;
	Vector4f curr3Dpoint, corr3Dnormal; Vector2f tmp2Dpoint;

	tmp3Dpoint.x = depth * ((float(x) - viewIntrinsics.z) / viewIntrinsics.x);
	tmp3Dpoint.y = depth * ((float(y) - viewIntrinsics.w) / viewIntrinsics.y);
	tmp3Dpoint.z = depth;
	tmp3Dpoint.w = 1.0f;

	// transform to previous frame coordinates
	tmp3Dpoint = approxInvPose * tmp3Dpoint;
	tmp3Dpoint.w = 1.0f;

	// project into previous rendered image
	tmp3Dpoint_reproj = scenePose * tmp3Dpoint;
	if (tmp3Dpoint_reproj.z <= 0.0f) return false;
	tmp2Dpoint.x = sceneIntrinsics.x * tmp3Dpoint_reproj.x / tmp3Dpoint_reproj.z + sceneIntrinsics.z;
	tmp2Dpoint.y = sceneIntrinsics.y * tmp3Dpoint_reproj.y / tmp3Dpoint_reproj.z + sceneIntrinsics.w;

	if (!((tmp2Dpoint.x >= 0.0f) && (tmp2Dpoint.x <= sceneImageSize.x - 2) && (tmp2Dpoint.y >= 0.0f) && (tmp2Dpoint.y <= sceneImageSize.y - 2)))
		return false;

	curr3Dpoint = interpolateBilinear_withHoles(pointsMap, tmp2Dpoint, sceneImageSize);
	if (curr3Dpoint.w < 0.0f) return false;

	ptDiff.x = curr3Dpoint.x - tmp3Dpoint.x;
	ptDiff.y = curr3Dpoint.y - tmp3Dpoint.y;
	ptDiff.z = curr3Dpoint.z - tmp3Dpoint.z;

	error = ORUtils::length(ptDiff);
	if (error != error) // check NaN
		return false;

	return true;
}

/**
 * Compute Jacobian (A) and residual (residual) for intensity at the given coordinate
 */
template<bool shortIteration, bool rotationOnly>
_CPU_AND_GPU_CODE_ inline bool computePerPointGH_RGB_Ab(float* A, float& residual,
                                                        const int x, const int y,
                                                        const Vector4f* points_current,
                                                        const float* intensities_current,
                                                        const float* intensities_reference,
                                                        const Vector2f* gradients_reference,
                                                        const Vector2i imgSize_depth,
                                                        const Vector2i imgSize_rgb,
                                                        const Vector4f& intrinsics_depth,
                                                        const Vector4f& intrinsics_rgb,
                                                        const Matrix4f& approxInvPose,
                                                        const Matrix4f& intensityReferencePose,
                                                        const float viewFrustum_max,
                                                        const float intensityThresh,
                                                        const float minGradient)
{
	if (x >= imgSize_depth.x || y >= imgSize_depth.y) return false;
	int idx = x + y * imgSize_depth.x;

	const Vector4f point_current = points_current[idx];
	const float intensity_current = intensities_current[idx];

	if (point_current.w < 0.f || intensity_current < 0.f || point_current.z < 1e-3f || point_current.z > viewFrustum_max) return false;

	const Vector3f point_world = (approxInvPose * point_current).toVector3();
	const Vector3f point_reference = intensityReferencePose * point_world;

	if (point_reference.z <= 0) return false;

	// Project the point in the reference intensity frame
	const Vector2f point_reference_proj = project(point_reference, intrinsics_rgb);

	// Outside the image plane
	if (point_reference_proj.x < 0 || point_reference_proj.x >= imgSize_rgb.x - 1 ||
	    point_reference_proj.y < 0 || point_reference_proj.y >= imgSize_rgb.y - 1) return false;

	const float intensity_reference = interpolateBilinear_single(intensities_reference, point_reference_proj, imgSize_rgb);
	const Vector2f gradient_reference = interpolateBilinear_Vector2(gradients_reference, point_reference_proj, imgSize_rgb);

	float diff = intensity_reference - intensity_current;
	if (fabs(diff) > intensityThresh) return false;
	if (fabs(gradient_reference.x) < minGradient || fabs(gradient_reference.y) < minGradient) return false;
	residual = diff;

	MatrixXf<2, 3> d_proj;
	const float inv_z = 1 / point_reference.z;
	const float inv_z_sq = inv_z * inv_z;
	d_proj(0, 0) = inv_z;
	d_proj(1, 0) = 0;
	d_proj(2, 0) = - point_reference.x * inv_z_sq;
	d_proj(0, 1) = 0;
	d_proj(1, 1) = d_proj(0, 0);
	d_proj(2, 1) = - point_reference.y * inv_z_sq;

	MatrixXf<1, 2> d_grad;
	d_grad(0, 0) = gradient_reference.x;
	d_grad(1, 0) = gradient_reference.y;

	MatrixXf<3, 3> K;
	K.setZeros();
	K(0, 0) = intrinsics_depth.x;
	K(1, 1) = intrinsics_depth.y;
	K(2, 0) = intrinsics_depth.z;
	K(2, 1) = intrinsics_depth.w;
	K(2, 2) = 1;

	if (shortIteration)
	{
		MatrixXf<3, 3> d_point;
		d_point.setZeros();
		if (rotationOnly)
		{
			d_point(0, 0) = 0;
			d_point(0, 1) = -point_world.z;
			d_point(0, 2) = point_world.y;
			d_point(1, 0) = point_world.z;
			d_point(1, 1) = 0;
			d_point(1, 2) = -point_world.x;
			d_point(2, 0) = -point_world.y;
			d_point(2, 1) = point_world.x;
			d_point(2, 2) = 0;
		}
		else
		{
			d_point(0, 0) = 1;
			d_point(1, 1) = 1;
			d_point(2, 2) = 1;
		}
		MatrixXf<1, 3> J = d_grad * d_proj * K * d_point;
		memcpy(A, J.m, sizeof(J.m));
	}
	else
	{
		MatrixXf<3, 6> d_point;
		d_point.setZeros();
		d_point(0, 0) = 0;
		d_point(0, 1) = -point_world.z;
		d_point(0, 2) = point_world.y;
		d_point(1, 0) = point_world.z;
		d_point(1, 1) = 0;
		d_point(1, 2) = -point_world.x;
		d_point(2, 0) = -point_world.y;
		d_point(2, 1) = point_world.x;
		d_point(2, 2) = 0;

		d_point(3, 0) = 1;
		d_point(4, 1) = 1;
		d_point(5, 2) = 1;

		MatrixXf<1, 6> J = d_grad * d_proj * K * d_point;
		memcpy(A, J.m, sizeof(J.m));
	}

	return true;
}

/**
 * Compute Jacobian (A) and residual (b) for depth point at the given coordinate
 */
template<bool shortIteration, bool rotationOnly>
_CPU_AND_GPU_CODE_ inline bool computePerPointGH_Depth_Ab(THREADPTR(float) *A, THREADPTR(float) &b,
	const THREADPTR(int) & x, const THREADPTR(int) & y,
	const CONSTPTR(float) &depth, const CONSTPTR(Vector2i) & viewImageSize, const CONSTPTR(Vector4f) & viewIntrinsics, const CONSTPTR(Vector2i) & sceneImageSize,
	const CONSTPTR(Vector4f) & sceneIntrinsics, const CONSTPTR(Matrix4f) & approxInvPose, const CONSTPTR(Matrix4f) & scenePose, const CONSTPTR(Vector4f) *pointsMap,
	const CONSTPTR(Vector4f) *normalsMap, float distThresh)
{
	if (depth <= 1e-8f) return false; //check if valid -- != 0.0f

	Vector4f tmp3Dpoint, tmp3Dpoint_reproj; Vector3f ptDiff;
	Vector4f curr3Dpoint, corr3Dnormal; Vector2f tmp2Dpoint;

	tmp3Dpoint.x = depth * ((float(x) - viewIntrinsics.z) / viewIntrinsics.x);
	tmp3Dpoint.y = depth * ((float(y) - viewIntrinsics.w) / viewIntrinsics.y);
	tmp3Dpoint.z = depth;
	tmp3Dpoint.w = 1.0f;
    
	// transform to previous frame coordinates
	tmp3Dpoint = approxInvPose * tmp3Dpoint;
	tmp3Dpoint.w = 1.0f;

	// project into previous rendered image
	tmp3Dpoint_reproj = scenePose * tmp3Dpoint;
	if (tmp3Dpoint_reproj.z <= 0.0f) return false;
	tmp2Dpoint.x = sceneIntrinsics.x * tmp3Dpoint_reproj.x / tmp3Dpoint_reproj.z + sceneIntrinsics.z;
	tmp2Dpoint.y = sceneIntrinsics.y * tmp3Dpoint_reproj.y / tmp3Dpoint_reproj.z + sceneIntrinsics.w;

	if (!((tmp2Dpoint.x >= 0.0f) && (tmp2Dpoint.x <= sceneImageSize.x - 2) && (tmp2Dpoint.y >= 0.0f) && (tmp2Dpoint.y <= sceneImageSize.y - 2)))
		return false;

	curr3Dpoint = interpolateBilinear_withHoles(pointsMap, tmp2Dpoint, sceneImageSize);
	if (curr3Dpoint.w < 0.0f) return false;

	ptDiff.x = curr3Dpoint.x - tmp3Dpoint.x;
	ptDiff.y = curr3Dpoint.y - tmp3Dpoint.y;
	ptDiff.z = curr3Dpoint.z - tmp3Dpoint.z;
	float dist = ptDiff.x * ptDiff.x + ptDiff.y * ptDiff.y + ptDiff.z * ptDiff.z;

	if (dist > distThresh or dist != dist) return false;

	corr3Dnormal = interpolateBilinear_withHoles(normalsMap, tmp2Dpoint, sceneImageSize);
//	if (corr3Dnormal.w < 0.0f) return false;

	b = ORUtils::dot(corr3Dnormal.toVector3(), ptDiff);

	// TODO check whether normal matches normal from image, done in the original paper, but does not seem to be required
	if (shortIteration)
	{
		if (rotationOnly)
		{
			Vector3f XP = ORUtils::cross(corr3Dnormal.toVector3(), tmp3Dpoint.toVector3());
			A[0] = XP.x;
			A[1] = XP.y;
			A[2] = XP.z;
		}
		else { A[0] = corr3Dnormal.x; A[1] = corr3Dnormal.y; A[2] = corr3Dnormal.z; }
	}
	else
	{
		Vector3f XP = ORUtils::cross(corr3Dnormal.toVector3(), tmp3Dpoint.toVector3());
		A[0] = XP.x;
		A[1] = XP.y;
		A[2] = XP.z;
		A[3] = corr3Dnormal.x;
		A[4] = corr3Dnormal.y;
		A[5] = corr3Dnormal.z;
	}

	return true;
}

template<bool shortIteration, bool rotationOnly>
_CPU_AND_GPU_CODE_ inline bool computePerPointGH_Depth(THREADPTR(float) *localNabla, THREADPTR(float) *localHessian, THREADPTR(float) &localF,
	const THREADPTR(int) & x, const THREADPTR(int) & y,
	const CONSTPTR(float) &depth, const CONSTPTR(Vector2i) & viewImageSize, const CONSTPTR(Vector4f) & viewIntrinsics, const CONSTPTR(Vector2i) & sceneImageSize,
	const CONSTPTR(Vector4f) & sceneIntrinsics, const CONSTPTR(Matrix4f) & approxInvPose, const CONSTPTR(Matrix4f) & scenePose, const CONSTPTR(Vector4f) *pointsMap,
	const CONSTPTR(Vector4f) *normalsMap, float distThresh)
{
	const int noPara = shortIteration ? 3 : 6;
	float A[noPara];
	float b;

	bool ret = computePerPointGH_Depth_Ab<shortIteration,rotationOnly>(A, b, x, y, depth, viewImageSize, viewIntrinsics, sceneImageSize, sceneIntrinsics, approxInvPose, scenePose, pointsMap, normalsMap, distThresh);

	if (!ret) return false;

	localF = b * b;

#if (defined(__CUDACC__) && defined(__CUDA_ARCH__))
#pragma unroll
#endif
	for (int r = 0, counter = 0; r < noPara; r++)
	{
		localNabla[r] = b * A[r];
#if (defined(__CUDACC__) && defined(__CUDA_ARCH__))
#pragma unroll
#endif
		for (int c = 0; c <= r; c++, counter++) localHessian[counter] = A[r] * A[c];
	}

	return true;
}

}
