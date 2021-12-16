// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <Utils/ITMPixelUtils.h>
#include <ITMLib/Utils/ITMProjectionUtils.h>

namespace ITMLib
{

/**
 * Project point form depth image view into (rendered) scene view, resulting in 2D coordinates
 * @param sceneCoords result 2D coordinates for scene point map
 * @param depthPoint_view depth point in depth view frame
 * @param approxInvPose assumed pose of current view (basis for projective DA hypothesis)
 * @param scenePose pose from which scene was rendered
 */
_CPU_AND_GPU_CODE_ inline bool projectiveDataAssociation(Vector2f& sceneCoords,
                                                         const Vector4f& depthPoint_view,
                                                         const Vector2i& viewImageSize,
                                                         const Vector4f& viewIntrinsics,
                                                         const Vector2i& sceneImageSize,
                                                         const Vector4f& sceneIntrinsics,
                                                         const Matrix4f& approxInvPose,
                                                         const Matrix4f& scenePose)
{
	Vector4f depthPoint_world = approxInvPose * depthPoint_view;

	// project into previous rendered image (scene)
	Vector4f depthPoint_scene = scenePose * depthPoint_world;
	if (depthPoint_scene.z <= 0.0f) return false;
	Vector2f depthPoint2DCoords = project(depthPoint_scene.toVector3(), sceneIntrinsics);

	if (!((depthPoint2DCoords.x >= 0.0f) && (depthPoint2DCoords.x <= sceneImageSize.x - 2) &&
	      (depthPoint2DCoords.y >= 0.0f) &&
	      (depthPoint2DCoords.y <= sceneImageSize.y - 2)))
		return false;

	sceneCoords = depthPoint2DCoords;

	return true;
}

template<bool shortIteration, bool rotationOnly>
_CPU_AND_GPU_CODE_ inline bool computePerPointError(float& error,
                                                    const int& x, const int& y,
                                                    const float* depth, const Vector2i& viewImageSize,
                                                    const Vector4f& viewIntrinsics, const Vector2i& sceneImageSize,
                                                    const Vector4f& sceneIntrinsics, const Matrix4f& approxInvPose,
                                                    const Matrix4f& scenePose, const Vector4f* pointsMap)
{
	float depthValue = depth[PixelCoordsToIndex(x, y, viewImageSize)];
	if (depthValue <= 1e-8f) return false; //check if valid -- != 0.0f
	Vector4f depthPoint_view = Vector4f(reprojectImagePoint(x, y, depthValue, invertProjectionParams(viewIntrinsics)), 1);

	Vector2f sceneCoords;
	if (!projectiveDataAssociation(sceneCoords, depthPoint_view, viewImageSize, viewIntrinsics,
	                               sceneImageSize, sceneIntrinsics, approxInvPose, scenePose))
		return false;
	Vector3f depthPoint_world = (approxInvPose * depthPoint_view).toVector3();

	Vector4f scenePoint_world = interpolateBilinear_withHoles(pointsMap, sceneCoords, sceneImageSize);
	if (scenePoint_world.w < 0.0f) return false;

	Vector3f ptDiff = scenePoint_world.toVector3() - depthPoint_world;
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
	int idx = PixelCoordsToIndex(x, y, imgSize_depth);

	const Vector4f point_current = points_current[idx];
	const float intensity_current = intensities_current[idx];

	if (point_current.w < 0.f || intensity_current < 0.f || point_current.z < 1e-3f ||
	    point_current.z > viewFrustum_max)
		return false;

	const Vector3f point_world = (approxInvPose * point_current).toVector3();
	const Vector3f point_reference = intensityReferencePose * point_world;

	if (point_reference.z <= 0) return false;

	// Project the point in the reference intensity frame
	const Vector2f point_reference_proj = project(point_reference, intrinsics_rgb);

	// Outside the image plane
	if (point_reference_proj.x < 0 || point_reference_proj.x >= imgSize_rgb.x - 1 ||
	    point_reference_proj.y < 0 || point_reference_proj.y >= imgSize_rgb.y - 1)
		return false;

	const float intensity_reference = interpolateBilinear_single(intensities_reference, point_reference_proj,
	                                                             imgSize_rgb);
	const Vector2f gradient_reference = interpolateBilinear_Vector2(gradients_reference, point_reference_proj,
	                                                                imgSize_rgb);

	float diff = intensity_reference - intensity_current;
	if (fabs(diff) > intensityThresh) return false;
	if (fabs(gradient_reference.x) < minGradient || fabs(gradient_reference.y) < minGradient) return false;
	residual = diff;

	// detailed derivative
	// d grad   d proj   d K   d T * p
	// ------ * ------ * --- * -------
	// d proj     d K    d p     d Xi

//	const float inv_z = 1 / point_reference.z;
//	const float inv_z_sq = inv_z * inv_z;
//	MatrixXf<2, 3> d_proj;
//	d_proj(0, 0) = inv_z;
//	d_proj(1, 0) = 0;
//	d_proj(2, 0) = -(point_reference.x * intrinsics_rgb.x + intrinsics_rgb.z * point_reference.z) * inv_z_sq;
//	d_proj(0, 1) = 0;
//	d_proj(1, 1) = inv_z;
//	d_proj(2, 1) = -(point_reference.y * intrinsics_rgb.y + intrinsics_rgb.w * point_reference.z) * inv_z_sq;
//
//
//	MatrixXf<1, 2> d_grad;
//	d_grad(0, 0) = gradient_reference.x;
//	d_grad(1, 0) = gradient_reference.y;
//
//	MatrixXf<3, 3> K;
//	K.setZeros();
//	K(0, 0) = intrinsics_rgb.x;
//	K(1, 1) = intrinsics_rgb.y;
//	K(2, 0) = intrinsics_rgb.z;
//	K(2, 1) = intrinsics_rgb.w;
//	K(2, 2) = 1;
//
//	if (shortIteration)
//	{
//		MatrixXf<3, 3> d_point;
//		d_point.setZeros();
//		if (rotationOnly)
//		{
//			d_point(0, 0) = 0;
//			d_point(0, 1) = -point_reference.z;
//			d_point(0, 2) = point_reference.y;
//			d_point(1, 0) = point_reference.z;
//			d_point(1, 1) = 0;
//			d_point(1, 2) = -point_reference.x;
//			d_point(2, 0) = -point_reference.y;
//			d_point(2, 1) = point_reference.x;
//			d_point(2, 2) = 0;
//		} else
//		{
//			d_point(0, 0) = 1;
//			d_point(1, 1) = 1;
//			d_point(2, 2) = 1;
//		}
//		MatrixXf<1, 3> J = d_grad * d_proj * K * d_point;
//		memcpy(A, J.m, sizeof(J.m));
//	} else
//	{
//		MatrixXf<3, 6> d_point;
//		d_point.setZeros();
//		d_point(0, 0) = 0;
//		d_point(0, 1) = -point_reference.z;
//		d_point(0, 2) = point_reference.y;
//		d_point(1, 0) = point_reference.z;
//		d_point(1, 1) = 0;
//		d_point(1, 2) = -point_reference.x;
//		d_point(2, 0) = -point_reference.y;
//		d_point(2, 1) = point_reference.x;
//		d_point(2, 2) = 0;
//
//		d_point(3, 0) = 1;
//		d_point(4, 1) = 1;
//		d_point(5, 2) = 1;
//
//		MatrixXf<1, 6> J = d_grad * d_proj * K * d_point;
//		memcpy(A, J.m, sizeof(J.m));
//  }

	// compact form of the above
	const float inv_z = 1 / point_reference.z;
	Vector3f dI_dproj_dK;
	dI_dproj_dK.x = gradient_reference.x * intrinsics_rgb.x * inv_z;
	dI_dproj_dK.y = gradient_reference.y * intrinsics_rgb.y * inv_z;
	dI_dproj_dK.z = -(dI_dproj_dK.x * point_reference.x + dI_dproj_dK.y * point_reference.y) * inv_z;

	if (shortIteration)
	{
		if (rotationOnly)
		{
			A[0] = -point_reference.z * dI_dproj_dK.y + point_reference.y * dI_dproj_dK.z;
			A[1] = point_reference.z * dI_dproj_dK.x - point_reference.x * dI_dproj_dK.z;
			A[2] = -point_reference.y * dI_dproj_dK.x + point_reference.x * dI_dproj_dK.y;
		} else
		{
			A[0] = dI_dproj_dK.x;
			A[1] = dI_dproj_dK.y;
			A[2] = dI_dproj_dK.z;
		}
	} else
	{
		A[0] = -point_reference.z * dI_dproj_dK.y + point_reference.y * dI_dproj_dK.z;
		A[1] = point_reference.z * dI_dproj_dK.x - point_reference.x * dI_dproj_dK.z;
		A[2] = -point_reference.y * dI_dproj_dK.x + point_reference.x * dI_dproj_dK.y;
		A[3] = dI_dproj_dK.x;
		A[4] = dI_dproj_dK.y;
		A[5] = dI_dproj_dK.z;
	}

	return true;
}

/**
 * Compute Jacobian (A) and residual (b) for depth point at the given coordinate
 */
template<bool shortIteration, bool rotationOnly>
_CPU_AND_GPU_CODE_ inline bool computePerPointGH_Depth_Ab(float* A, float& b,
                                                          const int& x, const int& y,
                                                          const float* depth, const Vector2i& viewImageSize,
                                                          const Vector4f& viewIntrinsics,
                                                          const Vector2i& sceneImageSize,
                                                          const Vector4f& sceneIntrinsics,
                                                          const Matrix4f& approxInvPose, const Matrix4f& scenePose,
                                                          const Vector4f* pointsMap,
                                                          const Vector4f* normalsMap,
                                                          float distThresh)
{
	float depthValue = depth[PixelCoordsToIndex(x, y, viewImageSize)];
	if (depthValue <= 1e-8f) return false; //check if valid -- != 0.0f
	Vector4f depthPoint_view = Vector4f(reprojectImagePoint(x, y, depthValue, invertProjectionParams(viewIntrinsics)), 1);

	Vector2f sceneCoords;
	if (!projectiveDataAssociation(sceneCoords, depthPoint_view, viewImageSize, viewIntrinsics,
	                               sceneImageSize, sceneIntrinsics, approxInvPose, scenePose))
		return false;
	Vector3f depthPoint_world = (approxInvPose * depthPoint_view).toVector3();

	Vector4f scenePoint_world = interpolateBilinear_withHoles(pointsMap, sceneCoords, sceneImageSize);
	if (scenePoint_world.w < 0.0f) return false;

	Vector3f ptDiff = scenePoint_world.toVector3() - depthPoint_world;
	float dist = ORUtils::dot(ptDiff, ptDiff);
	if (dist > distThresh or dist != dist) return false;

	Vector3f sceneNormal = interpolateBilinear_withHoles(normalsMap, sceneCoords, sceneImageSize).toVector3();
//	if (sceneNormal.w < 0.0f) return false;

	b = ORUtils::dot(sceneNormal, ptDiff);

	// TODO check whether normal matches normal from image, done in the original paper, but does not seem to be required
	if (shortIteration)
	{
		if (rotationOnly)
		{
			Vector3f XP = ORUtils::cross(depthPoint_world, sceneNormal);
			A[0] = -XP.x;
			A[1] = -XP.y;
			A[2] = -XP.z;
		} else
		{
			A[0] = -sceneNormal.x;
			A[1] = -sceneNormal.y;
			A[2] = -sceneNormal.z;
		}
	} else
	{
		Vector3f XP = ORUtils::cross(depthPoint_world, sceneNormal);
		A[0] = -XP.x;
		A[1] = -XP.y;
		A[2] = -XP.z;
		A[3] = -sceneNormal.x;
		A[4] = -sceneNormal.y;
		A[5] = -sceneNormal.z;
	}

	return true;
}

/**
 * Computes derivative for sim(3) point-to-plane Depth ICP
 * @param J 7-vector partial derivative of error function wrt. sim(3) parameters
 * @param r residual (value of error function)
 * @param x
 * @param y
 * @param depth
 * @param viewImageSize
 * @param viewIntrinsics
 * @param sceneImageSize
 * @param sceneIntrinsics
 * @param approxInvPose
 * @param scenePose
 * @param pointsMap
 * @param normalsMap
 * @param distThresh
 * @return
 */
_CPU_AND_GPU_CODE_ inline bool computeSim3Derivative_Depth(MatrixXf<1, 7>& J, float& r,
                                                           const int& x, const int& y,
                                                           const float* depth, const Vector2i& viewImageSize,
                                                           const Vector4f& viewIntrinsics,
                                                           const Vector2i& sceneImageSize,
                                                           const Vector4f& sceneIntrinsics,
                                                           const Matrix4f& approxInvPose, const Matrix4f& scenePose,
                                                           const Vector4f* pointsMap,
                                                           const Vector4f* normalsMap,
                                                           float distThresh)
{
	float depthValue = depth[PixelCoordsToIndex(x, y, viewImageSize)];
	if (depthValue <= 1e-8f) return false; //check if valid -- != 0.0f
	Vector4f depthPoint_view = Vector4f(reprojectImagePoint(x, y, depthValue, invertProjectionParams(viewIntrinsics)), 1);

	Vector2f sceneCoords;
	if (!projectiveDataAssociation(sceneCoords, depthPoint_view, viewImageSize, viewIntrinsics,
	                               sceneImageSize, sceneIntrinsics, approxInvPose, scenePose))
		return false;
	Vector3f depthPoint_world = (approxInvPose * depthPoint_view).toVector3();

	Vector4f scenePoint_world = interpolateBilinear_withHoles(pointsMap, sceneCoords, sceneImageSize);
	if (scenePoint_world.w < 0.0f) return false;

	Vector3f ptDiff = scenePoint_world.toVector3() - depthPoint_world;
	float dist = ORUtils::dot(ptDiff, ptDiff);
	if (dist > distThresh or dist != dist) return false;

	Vector3f sceneNormal = interpolateBilinear_withHoles(normalsMap, sceneCoords, sceneImageSize).toVector3();

	r = ORUtils::dot(sceneNormal, ptDiff);

	Vector3f XP = ORUtils::cross(depthPoint_world, sceneNormal);
	J(0, 0) = -sceneNormal.x;
	J(1, 0) = -sceneNormal.y;
	J(2, 0) = -sceneNormal.z;
	J(3, 0) = -XP.x;
	J(4, 0) = -XP.y;
	J(5, 0) = -XP.z;
	J(6, 0) = -ORUtils::dot(sceneNormal, depthPoint_world);

	return true;
}

/**
 * Computes derivative for sim(3) point-to-plane RGB ICP
 * @param J 7-vector partial derivative of error function wrt. sim(3) parameters
 * @param r residual (value of error function)
 * @return
 */
_CPU_AND_GPU_CODE_ inline bool computeSim3Derivative_RGB(MatrixXf<1, 7>& J, float& r,
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
	int idx = PixelCoordsToIndex(x, y, imgSize_depth);

	const Vector4f point_current = points_current[idx];
	const float intensity_current = intensities_current[idx];

	if (point_current.w < 0.f || intensity_current < 0.f || point_current.z < 1e-3f ||
	    point_current.z > viewFrustum_max)
		return false;

	const Vector3f point_world = (approxInvPose * point_current).toVector3();
	const Vector3f point_reference = intensityReferencePose * point_world;

	if (point_reference.z <= 0) return false;

	// Project the point in the reference intensity frame
	const Vector2f point_reference_proj = project(point_reference, intrinsics_rgb);

	// Outside the image plane
	if (point_reference_proj.x < 0 || point_reference_proj.x >= imgSize_rgb.x - 1 ||
	    point_reference_proj.y < 0 || point_reference_proj.y >= imgSize_rgb.y - 1)
		return false;

	const float intensity_reference = interpolateBilinear_single(intensities_reference, point_reference_proj,
	                                                             imgSize_rgb);
	const Vector2f gradient_reference = interpolateBilinear_Vector2(gradients_reference, point_reference_proj,
	                                                                imgSize_rgb);

	float diff = intensity_reference - intensity_current;
	if (fabs(diff) > intensityThresh) return false;
	if (fabs(gradient_reference.x) < minGradient || fabs(gradient_reference.y) < minGradient) return false;
	r = diff;

	MatrixXf<2, 3> d_proj;
	const float inv_z = 1 / point_reference.z;
	const float inv_z_sq = inv_z * inv_z;
	d_proj(0, 0) = inv_z;
	d_proj(1, 0) = 0;
	d_proj(2, 0) = -(point_reference.x * intrinsics_rgb.x + intrinsics_rgb.z * point_reference.z) * inv_z_sq;
	d_proj(0, 1) = 0;
	d_proj(1, 1) = inv_z;
	d_proj(2, 1) = -(point_reference.y * intrinsics_rgb.y + intrinsics_rgb.w * point_reference.z) * inv_z_sq;

	MatrixXf<1, 2> d_grad;
	d_grad(0, 0) = gradient_reference.x;
	d_grad(1, 0) = gradient_reference.y;

	MatrixXf<3, 3> K;
	K.setZeros();
	K(0, 0) = intrinsics_rgb.x;
	K(1, 1) = intrinsics_rgb.y;
	K(2, 0) = intrinsics_rgb.z;
	K(2, 1) = intrinsics_rgb.w;
	K(2, 2) = 1;

	MatrixXf<3, 7> d_point;
	d_point.setZeros();
	d_point(0, 0) = 1;
	d_point(1, 1) = 1;
	d_point(2, 2) = 1;

	d_point(3, 0) = 0;
	d_point(3, 1) = -point_reference.z;
	d_point(3, 2) = point_reference.y;
	d_point(4, 0) = point_reference.z;
	d_point(4, 1) = 0;
	d_point(4, 2) = -point_reference.x;
	d_point(5, 0) = -point_reference.y;
	d_point(5, 1) = point_reference.x;
	d_point(5, 2) = 0;

	d_point(6, 0) = point_world.x;
	d_point(6, 1) = point_world.y;
	d_point(6, 2) = point_world.z;

	J = d_grad * d_proj * K * d_point;
	return true;
}

template<bool shortIteration, bool rotationOnly>
_CPU_AND_GPU_CODE_ inline bool computePerPointGH_Depth(float* localNabla, float* localHessian, float& localF,
                                                       const int& x, const int& y,
                                                       const float* depth, const Vector2i& viewImageSize,
                                                       const Vector4f& viewIntrinsics, const Vector2i& sceneImageSize,
                                                       const Vector4f& sceneIntrinsics, const Matrix4f& approxInvPose,
                                                       const Matrix4f& scenePose, const Vector4f* pointsMap,
                                                       const Vector4f* normalsMap, float distThresh)
{
	const int noPara = shortIteration ? 3 : 6;
	float A[noPara];
	float b;

	bool ret = computePerPointGH_Depth_Ab<shortIteration, rotationOnly>(A, b, x, y, depth, viewImageSize, viewIntrinsics,
	                                                                    sceneImageSize, sceneIntrinsics, approxInvPose,
	                                                                    scenePose, pointsMap, normalsMap,
	                                                                    distThresh);

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
