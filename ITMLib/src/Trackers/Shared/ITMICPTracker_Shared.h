// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <Utils/ITMPixelUtils.h>
#include <ITMLib/Utils/ITMProjectionUtils.h>
#include "ITMTracker_Shared.h"

//#define USE_COLOR_WEIGHT
#define USE_DEPTH_WEIGHT

namespace ITMLib
{

/**
 * Project point form depth image view into (rendered) scene view, resulting in 2D coordinates
 * @param sceneCoords result 2D coordinates for scene point map
 * @param depthPoint_view depth point in depth view frame
 * @param deltaT current estimate of transform between current(view) frame and reference(scene) frame
 */
_CPU_AND_GPU_CODE_ inline bool projectiveDataAssociation(Vector2f& sceneCoords,
                                                         const Vector4f& depthPoint_view,
                                                         const Vector2i& viewImageSize,
                                                         const Vector4f& viewIntrinsics,
                                                         const Vector2i& sceneImageSize,
                                                         const Vector4f& sceneIntrinsics,
                                                         const Matrix4f& deltaT)
{
	Vector4f depthPoint_scene = deltaT * depthPoint_view;

	// project into previous rendered image (scene)
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
	                               sceneImageSize, sceneIntrinsics, scenePose * approxInvPose))
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
 *
 * @param deltaT current estimate of transform between current frame and reference frame
 * @param T_ref_CW transforms world point into reference frame
 */
template<bool shortIteration, bool rotationOnly>
_CPU_AND_GPU_CODE_ inline bool computePerPointGH_RGB_Ab(float* A, float& residual, float& weight,
                                                        const int x, const int y,
                                                        const Matrix4f& deltaT,
                                                        const Matrix4f& T_ref_CW,
                                                        const Matrix4f& T_scene_WC,
                                                        const Vector4f* points_current,
                                                        const float* intensities_current,
                                                        const float* intensities_reference,
                                                        const Vector2f* gradients_reference,
                                                        const Vector2i imgSize_depth,
                                                        const Vector2i imgSize_rgb,
                                                        const Vector4f& intrinsics_depth,
                                                        const Vector4f& intrinsics_rgb,
                                                        const float viewFrustum_max,
                                                        const float intensityThresh,
                                                        const float minGradient)
{
	if (x >= imgSize_depth.x || y >= imgSize_depth.y) return false;
	int idx = PixelCoordsToIndex(x, y, imgSize_depth);


	const Vector4f depthPoint_view = points_current[idx];
	const float intensity_current = intensities_current[idx];

	if (depthPoint_view.w < 0.f || intensity_current < 0.f || depthPoint_view.z < 1e-3f ||
	    depthPoint_view.z > viewFrustum_max)
		return false;

	const Vector3f depthPoint_scene = deltaT * depthPoint_view.toVector3();
	// transform point into intensity reference frame (deltaT is from current frame to rendered scene frame)
	const Matrix4f T_ReferenceScene = T_ref_CW * T_scene_WC;
//	Matrix4f T_ReferenceScene; T_ReferenceScene.setIdentity();
	const Vector3f depthPoint_reference = T_ReferenceScene * depthPoint_scene;

	if (depthPoint_reference.z <= 0) return false;

	// Project the depth point into the reference intensity frame
	const Vector2f imageCoords = project(depthPoint_reference, intrinsics_rgb);

	// Outside the image plane
	if (imageCoords.x < 0 || imageCoords.x >= imgSize_rgb.x - 1 ||
	    imageCoords.y < 0 || imageCoords.y >= imgSize_rgb.y - 1)
		return false;

	const float intensity_reference = interpolateBilinear_single(intensities_reference, imageCoords,
	                                                             imgSize_rgb);


	float diff = intensity_reference - intensity_current;
	Vector2f gradient_reference = interpolateBilinear_Vector2(gradients_reference, imageCoords,
	                                                          imgSize_rgb);
	if (fabs(diff) > intensityThresh) return false;
	if (fabs(gradient_reference.x) < minGradient || fabs(gradient_reference.y) < minGradient) return false;

	const float sobelScale = 1.0f / pow(2, 3); // from Kintinuous
	gradient_reference *= sobelScale;

	weight = 1;
#ifdef USE_COLOR_WEIGHT
	weight *= 0.05 * intensityThresh / MAX(fabs(diff), 0.001);
#endif

//	weight *= 50.0f * sqrt(ORUtils::dot(gradient_reference, gradient_reference));

#ifdef USE_DEPTH_WEIGHT
	weight *= CLAMP(1.0f / pow(depthPoint_scene.z + (1 - 0.1), 2), 0, 1);
//	weight *= CLAMP(1.0f / pow(depthPoint_scene.z, 2), 0, 1);
//		weight *= CLAMP(1.0f / depthPoint_scene.z, 0, 1);
#endif

	residual = diff;

	// detailed derivative
	// d grad   d proj   d K   d T * p
	// ------ * ------ * --- * -------
	// d proj     d K    d p     d Xi

//	const float inv_z = 1 / depthPoint_reference.z;
//	const float inv_z_sq = inv_z * inv_z;
//	MatrixXf<2, 3> d_proj;
//	d_proj(0, 0) = inv_z;
//	d_proj(1, 0) = 0;
//	d_proj(2, 0) = -depthPoint_reference.x * inv_z_sq;
//	d_proj(0, 1) = 0;
//	d_proj(1, 1) = inv_z;
//	d_proj(2, 1) = -depthPoint_reference.y * inv_z_sq;
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
//			d_point(0, 1) = -depthPoint_reference.z;
//			d_point(0, 2) = depthPoint_reference.y;
//			d_point(1, 0) = depthPoint_reference.z;
//			d_point(1, 1) = 0;
//			d_point(1, 2) = -depthPoint_reference.x;
//			d_point(2, 0) = -depthPoint_reference.y;
//			d_point(2, 1) = depthPoint_reference.x;
//			d_point(2, 2) = 0;
//		} else
//		{
//			d_point(0, 0) = 1;
//			d_point(1, 1) = 1;
//			d_point(2, 2) = 1;
//		}
//		MatrixXf<3, 3> R;
//		memcpy(R.m, T_ReferenceScene.getRotationMatrix().m, sizeof(R.m));
//
//		MatrixXf<1, 3> J = d_grad * d_proj * K * R * d_point;
//		memcpy(A, J.m, sizeof(J.m));
//	} else
//	{
//		MatrixXf<3, 6> d_point;
//		d_point.setZeros();
//		d_point(0, 0) = 1;
//		d_point(1, 1) = 1;
//		d_point(2, 2) = 1;
//
//		d_point(3, 0) = 0;
//		d_point(3, 1) = -depthPoint_scene.z;
//		d_point(3, 2) = depthPoint_scene.y;
//		d_point(4, 0) = depthPoint_scene.z;
//		d_point(4, 1) = 0;
//		d_point(4, 2) = -depthPoint_scene.x;
//		d_point(5, 0) = -depthPoint_scene.y;
//		d_point(5, 1) = depthPoint_scene.x;
//		d_point(5, 2) = 0;
//
//		MatrixXf<3, 3> R;
//		memcpy(R.m, T_ReferenceScene.getRotationMatrix().m, sizeof(R.m));
//
//		MatrixXf<1, 6> J = d_grad * d_proj * K * R * d_point;
//		memcpy(A, J.m, sizeof(J.m));
//  }


	// compact form of the above
	const float inv_z = 1 / depthPoint_reference.z;
	MatrixXf<1, 3> dI_dproj_dK;
	dI_dproj_dK(0, 0) = gradient_reference.x * intrinsics_rgb.x * inv_z;
	dI_dproj_dK(1, 0) = gradient_reference.y * intrinsics_rgb.y * inv_z;
	dI_dproj_dK(2, 0) = gradient_reference.x * (intrinsics_rgb.z * inv_z - depthPoint_reference.x * inv_z * inv_z)
	                    + gradient_reference.y * (intrinsics_rgb.w * inv_z - depthPoint_reference.y * inv_z * inv_z);

	MatrixXf<3, 3> R;
	memcpy(R.m, T_ReferenceScene.getRotationMatrix().m, sizeof(R.m));
	dI_dproj_dK = dI_dproj_dK * R;

	if (shortIteration)
	{
		if (rotationOnly)
		{
			A[0] = -depthPoint_scene.z * dI_dproj_dK(1, 0) + depthPoint_scene.y * dI_dproj_dK(2, 0);
			A[1] = depthPoint_scene.z * dI_dproj_dK(0, 0) - depthPoint_scene.x * dI_dproj_dK(2, 0);
			A[2] = -depthPoint_scene.y * dI_dproj_dK(0, 0) + depthPoint_scene.x * dI_dproj_dK(1, 0);
		} else
		{
			A[0] = dI_dproj_dK(0, 0);
			A[1] = dI_dproj_dK(1, 0);
			A[2] = dI_dproj_dK(2, 0);
		}
	} else
	{
		A[0] = dI_dproj_dK(0, 0);
		A[1] = dI_dproj_dK(1, 0);
		A[2] = dI_dproj_dK(2, 0);
		A[3] = -depthPoint_scene.z * dI_dproj_dK(1, 0) + depthPoint_scene.y * dI_dproj_dK(2, 0);
		A[4] = depthPoint_scene.z * dI_dproj_dK(0, 0) - depthPoint_scene.x * dI_dproj_dK(2, 0);
		A[5] = -depthPoint_scene.y * dI_dproj_dK(0, 0) + depthPoint_scene.x * dI_dproj_dK(1, 0);
	}

	return true;
}

/**
 * Compute Jacobian (A) and residual (b) for depth point at the given coordinate
 *
 * @param deltaT current estimate of transform between current frame and reference(scene) frame
 * @param T_scene_CW transforms world point into reference frame
 */
template<bool shortIteration, bool rotationOnly>
_CPU_AND_GPU_CODE_ inline bool computePerPointGH_Depth_Ab(float* A, float& b, float& weight,
                                                          const int& x, const int& y,
                                                          const Matrix4f& deltaT, const Matrix4f& T_scene_CW,
                                                          const float* depth, const Vector2i& viewImageSize,
                                                          const Vector4f& viewIntrinsics,
                                                          const Vector2i& sceneImageSize,
                                                          const Vector4f& sceneIntrinsics,
                                                          const Vector4f* pointsMap,
                                                          const Vector4f* normalsMap,
                                                          float distThresh)
{
	float depthValue = depth[PixelCoordsToIndex(x, y, viewImageSize)];
	if (depthValue <= 1e-8f) return false; //check if valid -- != 0.0f
	Vector4f depthPoint_view = Vector4f(reprojectImagePoint(x, y, depthValue, invertProjectionParams(viewIntrinsics)), 1);

	Vector2f sceneCoords;
	if (!projectiveDataAssociation(sceneCoords, depthPoint_view, viewImageSize, viewIntrinsics,
	                               sceneImageSize, sceneIntrinsics, deltaT))
		return false;
	Vector3f depthPoint_scene = (deltaT * depthPoint_view).toVector3();


	Vector4f scenePoint_world = interpolateBilinear_withHoles(pointsMap, sceneCoords, sceneImageSize);
	if (scenePoint_world.w < 0.0f) return false;

	Vector3f scenePoint_scene = T_scene_CW * scenePoint_world.toVector3();

	Vector3f ptDiff = scenePoint_scene - depthPoint_scene;
	float dist = ORUtils::dot(ptDiff, ptDiff);
	if (dist > distThresh or dist != dist) return false;

	Vector3f sceneNormal_world = interpolateBilinear_withHoles(normalsMap, sceneCoords, sceneImageSize).toVector3();
	Vector3f sceneNormal_scene = (T_scene_CW * Vector4f(sceneNormal_world, 0)).toVector3();
//	if (sceneNormal_world.w < 0.0f) return false;

	weight = 1;
#ifdef USE_DEPTH_WEIGHT
	weight *= CLAMP(1.0f / pow(depthPoint_scene.z + (1 - 0.1), 2), 0, 1);
//	weight *= 1.0f / pow(depthPoint_scene.z, 2);
//	weight *= CLAMP(1.0f / depthPoint_scene.z, 0, 1);
#endif

	b = ORUtils::dot(sceneNormal_scene, ptDiff);

	// TODO check whether normal matches normal from image, done in the original paper, but does not seem to be required
	if (shortIteration)
	{
		if (rotationOnly)
		{
			Vector3f XP = ORUtils::cross(depthPoint_scene, sceneNormal_scene);
			A[0] = -XP.x;
			A[1] = -XP.y;
			A[2] = -XP.z;
		} else
		{
			A[0] = -sceneNormal_scene.x;
			A[1] = -sceneNormal_scene.y;
			A[2] = -sceneNormal_scene.z;
		}
	} else
	{
		Vector3f XP = ORUtils::cross(depthPoint_scene, sceneNormal_scene);
		A[0] = -sceneNormal_scene.x;
		A[1] = -sceneNormal_scene.y;
		A[2] = -sceneNormal_scene.z;
		A[3] = -XP.x;
		A[4] = -XP.y;
		A[5] = -XP.z;
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
_CPU_AND_GPU_CODE_ inline bool computeSim3Derivative_Depth(MatrixXf<1, 7>& J, float& r, float& weight,
                                                           const int& x, const int& y,
                                                           const float* depth, const Vector2i& viewImageSize,
                                                           const Vector4f& viewIntrinsics,
                                                           const Vector2i& sceneImageSize,
                                                           const Vector4f& sceneIntrinsics,
                                                           const Matrix4f& deltaT, const Matrix4f& T_scene_CW,
                                                           const Vector4f* pointsMap,
                                                           const Vector4f* normalsMap,
                                                           float distThresh)
{
	float depthValue = depth[PixelCoordsToIndex(x, y, viewImageSize)];
	if (depthValue <= 1e-8f) return false; //check if valid -- != 0.0f
	Vector4f depthPoint_view = Vector4f(reprojectImagePoint(x, y, depthValue, invertProjectionParams(viewIntrinsics)), 1);

	Vector2f sceneCoords;
	if (!projectiveDataAssociation(sceneCoords, depthPoint_view, viewImageSize, viewIntrinsics,
	                               sceneImageSize, sceneIntrinsics, deltaT))
		return false;
	Vector3f depthPoint_scene = (deltaT * depthPoint_view).toVector3();


	Vector4f scenePoint_world = interpolateBilinear_withHoles(pointsMap, sceneCoords, sceneImageSize);
	if (scenePoint_world.w < 0.0f) return false;

	Vector3f scenePoint_scene = T_scene_CW * scenePoint_world.toVector3();

	Vector3f ptDiff = scenePoint_scene - depthPoint_scene;
	float dist = ORUtils::dot(ptDiff, ptDiff);
	if (dist > distThresh or dist != dist) return false;

	Vector3f sceneNormal_world = interpolateBilinear_withHoles(normalsMap, sceneCoords, sceneImageSize).toVector3();
	Vector3f sceneNormal_scene = (T_scene_CW * Vector4f(sceneNormal_world, 0)).toVector3();
//	if (sceneNormal_world.w < 0.0f) return false;

	weight = 1;
#ifdef USE_DEPTH_WEIGHT
	weight *= CLAMP(1.0f / pow(depthPoint_scene.z + (1 - 0.1), 2), 0, 1);
//	weight *= CLAMP(1.0f / pow(depthPoint_scene.z, 2), 0, 1);
//	weight *= CLAMP(1.0f / depthPoint_scene.z, 0, 1);
#endif

	r = ORUtils::dot(sceneNormal_scene, ptDiff);

	Vector3f XP = ORUtils::cross(depthPoint_scene, sceneNormal_scene);
	J(0, 0) = -sceneNormal_scene.x;
	J(1, 0) = -sceneNormal_scene.y;
	J(2, 0) = -sceneNormal_scene.z;
	J(3, 0) = -XP.x;
	J(4, 0) = -XP.y;
	J(5, 0) = -XP.z;
	J(6, 0) = -ORUtils::dot(sceneNormal_scene, depthPoint_scene);

	return true;
}

/**
 * Computes derivative for sim(3) point-to-plane RGB ICP
 * @param J 7-vector partial derivative of error function wrt. sim(3) parameters
 * @param r residual (value of error function)
 * @return
 */
_CPU_AND_GPU_CODE_ inline bool computeSim3Derivative_RGB(MatrixXf<1, 7>& J, float& r, float& weight,
                                                         const int x, const int y,
                                                         const Vector4f* points_current,
                                                         const float* intensities_current,
                                                         const float* intensities_reference,
                                                         const Vector2f* gradients_reference,
                                                         const Vector2i imgSize_depth,
                                                         const Vector2i imgSize_rgb,
                                                         const Vector4f& intrinsics_depth,
                                                         const Vector4f& intrinsics_rgb,
                                                         const Matrix4f& deltaT,
                                                         const Matrix4f& T_ref_CW,
                                                         const Matrix4f& T_scene_WC,
                                                         const float viewFrustum_max,
                                                         const float intensityThresh,
                                                         const float minGradient)
{
	if (x >= imgSize_depth.x || y >= imgSize_depth.y) return false;
	int idx = PixelCoordsToIndex(x, y, imgSize_depth);

	const Vector4f depthPoint_view = points_current[idx];
	const float intensity_current = intensities_current[idx];

	if (depthPoint_view.w < 0.f || intensity_current < 0.f || depthPoint_view.z < 1e-3f ||
	    depthPoint_view.z > viewFrustum_max)
		return false;

	const Vector3f depthPoint_scene = deltaT * depthPoint_view.toVector3();
	// transform point into intensity reference frame (deltaT is from current frame to rendered scene frame)
	const Matrix4f T_ReferenceScene = T_ref_CW * T_scene_WC;
	const Vector3f depthPoint_reference = T_ReferenceScene * depthPoint_scene;

	if (depthPoint_reference.z <= 0) return false;

	// Project the depth point into the reference intensity frame
	const Vector2f imageCoords = project(depthPoint_reference, intrinsics_rgb);

	// Outside the image plane
	if (imageCoords.x < 0 || imageCoords.x >= imgSize_rgb.x - 1 ||
	    imageCoords.y < 0 || imageCoords.y >= imgSize_rgb.y - 1)
		return false;

	const float intensity_reference = interpolateBilinear_single(intensities_reference, imageCoords,
	                                                             imgSize_rgb);


	float diff = intensity_reference - intensity_current;
	Vector2f gradient_reference = interpolateBilinear_Vector2(gradients_reference, imageCoords,
	                                                          imgSize_rgb);
	if (fabs(diff) > intensityThresh) return false;
	if (fabs(gradient_reference.x) < minGradient || fabs(gradient_reference.y) < minGradient) return false;

	const float sobelScale = 1.0f / pow(2, 3); // from Kintinuous
	gradient_reference *= sobelScale;

	weight = 1;
#ifdef USE_DEPTH_WEIGHT
	weight *= CLAMP(1.0f / pow(depthPoint_scene.z + (1 - 0.1), 2), 0, 1);
//	weight *= CLAMP(1.0f / pow(depthPoint_reference.z, 2), 0, 1);
//	weight *= CLAMP(1.0f / depthPoint_scene.z, 0, 1);
#endif

	r = diff;

	MatrixXf<2, 3> d_proj;
	const float inv_z = 1 / depthPoint_reference.z;
	const float inv_z_sq = inv_z * inv_z;
	d_proj(0, 0) = inv_z;
	d_proj(1, 0) = 0;
	d_proj(2, 0) = -depthPoint_reference.x * inv_z_sq;
	d_proj(0, 1) = 0;
	d_proj(1, 1) = inv_z;
	d_proj(2, 1) = -depthPoint_reference.y * inv_z_sq;

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
	d_point(3, 1) = -depthPoint_reference.z;
	d_point(3, 2) = depthPoint_reference.y;
	d_point(4, 0) = depthPoint_reference.z;
	d_point(4, 1) = 0;
	d_point(4, 2) = -depthPoint_reference.x;
	d_point(5, 0) = -depthPoint_reference.y;
	d_point(5, 1) = depthPoint_reference.x;
	d_point(5, 2) = 0;

	d_point(6, 0) = depthPoint_reference.x;
	d_point(6, 1) = depthPoint_reference.y;
	d_point(6, 2) = depthPoint_reference.z;

	J = d_grad * d_proj * K * d_point;
	return true;
}

template<bool shortIteration, bool rotationOnly>
_CPU_AND_GPU_CODE_ inline bool computePerPointGH_Depth(float* localNabla, float* localHessian, float& localF,
                                                       const int& x, const int& y,
                                                       const Matrix4f& deltaT, const Matrix4f& T_ref_CW,
                                                       const float* depth, const Vector2i& viewImageSize,
                                                       const Vector4f& viewIntrinsics, const Vector2i& sceneImageSize,
                                                       const Vector4f& sceneIntrinsics,
                                                       const Vector4f* pointsMap,
                                                       const Vector4f* normalsMap, float distThresh)
{
	const int noPara = shortIteration ? 3 : 6;
	float A[noPara];
	float b;
	float weight;

	bool ret = computePerPointGH_Depth_Ab<shortIteration, rotationOnly>(A, b, weight, x, y, deltaT, T_ref_CW, depth,
	                                                                    viewImageSize, viewIntrinsics,
	                                                                    sceneImageSize, sceneIntrinsics, pointsMap,
	                                                                    normalsMap,
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
