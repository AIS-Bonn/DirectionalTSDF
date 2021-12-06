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
                                                         const Matrix4f& scenePose,
                                                         const float scaleFactor = 1.0)
{
//	Vector4f depthPoint_world = approxInvPose * depthPoint_view;
	Vector4f depthPoint_world = (approxInvPose * Vector4f(scaleFactor * depthPoint_view.toVector3(), 0) +
	                             approxInvPose.getColumn(3));

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

	MatrixXf<2, 3> d_proj;
	const float inv_z = 1 / point_reference.z;
	const float inv_z_sq = inv_z * inv_z;
	d_proj(0, 0) = inv_z;
	d_proj(1, 0) = 0;
	d_proj(2, 0) = -point_reference.x * inv_z_sq;
	d_proj(0, 1) = 0;
	d_proj(1, 1) = d_proj(0, 0);
	d_proj(2, 1) = -point_reference.y * inv_z_sq;

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
		} else
		{
			d_point(0, 0) = 1;
			d_point(1, 1) = 1;
			d_point(2, 2) = 1;
		}
		MatrixXf<1, 3> J = d_grad * d_proj * K * d_point;
		memcpy(A, J.m, sizeof(J.m));
	} else
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
_CPU_AND_GPU_CODE_ inline bool computePerPointGH_Depth_Ab(float* A, float& b,
                                                          const int& x, const int& y,
                                                          const float* depth, const Vector2i& viewImageSize,
                                                          const Vector4f& viewIntrinsics,
                                                          const Vector2i& sceneImageSize,
                                                          const Vector4f& sceneIntrinsics,
                                                          const Matrix4f& approxInvPose, const Matrix4f& scenePose,
                                                          const Vector4f* pointsMap,
                                                          const Vector4f* normalsMap, float scaleFactor,
                                                          float distThresh)
{
	float depthValue = depth[PixelCoordsToIndex(x, y, viewImageSize)];
	if (depthValue <= 1e-8f) return false; //check if valid -- != 0.0f
	Vector4f depthPoint_view = Vector4f(reprojectImagePoint(x, y, depthValue, invertProjectionParams(viewIntrinsics)), 1);

	Vector2f sceneCoords;
	if (!projectiveDataAssociation(sceneCoords, depthPoint_view, viewImageSize, viewIntrinsics,
	                               sceneImageSize, sceneIntrinsics, approxInvPose, scenePose, scaleFactor))
		return false;
//	Vector3f depthPoint_world = (approxInvPose * depthPoint_view).toVector3();
	Vector3f depthPoint_world = (approxInvPose * Vector4f(scaleFactor * depthPoint_view.toVector3(), 0) +
	                             approxInvPose.getColumn(3)).toVector3();

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
 * Sahillioglu 2021: Scale-Adaptive ICP
 * @param f residual
 * @param a summand for upper left entry of matrix in Eq. (6)
 * @param d summand upper entry of vector on right side of Eq. (6)
 * @param c summand for vector c in Eq. (6)
 * @param d summand for vector d in Eq. (6)
 */
_CPU_AND_GPU_CODE_ inline bool computePerPoint_Sahillioglu(float& f, float& a, float& b, Vector3f& c, Vector3f& d,
                                                           const int& x, const int& y,
                                                           const float* depth, const Vector2i& viewImageSize,
                                                           const Vector4f& viewIntrinsics,
                                                           const Vector4f* pointsMap,
                                                           const Vector2i& sceneImageSize,
                                                           const Vector4f& sceneIntrinsics,
                                                           const Matrix4f& approxInvPose, const float approxScaleFactor,
                                                           const Matrix4f& scenePose,
                                                           float distThresh)
{
	float depthValue = depth[PixelCoordsToIndex(x, y, viewImageSize)];
	if (depthValue <= 1e-8f) return false; //check if valid -- != 0.0f
	Vector4f depthPoint_view = Vector4f(reprojectImagePoint(x, y, depthValue, invertProjectionParams(viewIntrinsics)), 1);

//	Vector2f sceneCoords;
//	if (!projectiveDataAssociation(sceneCoords, depthPoint_view, viewImageSize, viewIntrinsics,
//	                               sceneImageSize, sceneIntrinsics, approxInvPose, scenePose))
//		return false;

	// scale rotation only!
	Vector3f depthPoint_world = (approxInvPose * Vector4f(approxScaleFactor * depthPoint_view.toVector3(), 0) +
	                             approxInvPose.getColumn(3)).toVector3();

	// project into previous rendered image (scene)
	Vector4f depthPoint_scene = scenePose * Vector4f(depthPoint_world, 1);
	if (depthPoint_scene.z <= 0.0f) return false;
	Vector2f depthPoint2DCoords = project(depthPoint_scene.toVector3(), sceneIntrinsics);

	if (!((depthPoint2DCoords.x >= 0.0f) && (depthPoint2DCoords.x <= sceneImageSize.x - 2) &&
	      (depthPoint2DCoords.y >= 0.0f) &&
	      (depthPoint2DCoords.y <= sceneImageSize.y - 2)))
		return false;

	Vector2f sceneCoords = depthPoint2DCoords;

	// Corresponding point in rendered image (scene)
	Vector4f scenePoint_world = interpolateBilinear_withHoles(pointsMap, sceneCoords, sceneImageSize);
	if (scenePoint_world.w < 0.0f) return false;

	a = ORUtils::dot(depthPoint_world, depthPoint_world);
	b = ORUtils::dot(depthPoint_world, scenePoint_world.toVector3());
	c = depthPoint_world;
	d = scenePoint_world.toVector3();

	Vector3f ptDiff = scenePoint_world.toVector3() - depthPoint_world;
	f = ORUtils::dot(ptDiff, ptDiff);

	return true;
}


/**
 * Point-to-plane scale-translation optimization
 */
_CPU_AND_GPU_CODE_ inline bool computePerPoint_TransScale(float& f, Vector4f& j, float& r,
                                                          const int& x, const int& y,
                                                          const float* depth, const Vector2i& viewImageSize,
                                                          const Vector4f& viewIntrinsics,
                                                          const Vector4f* pointsMap,
                                                          const Vector4f* normalsMap,
                                                          const Vector2i& sceneImageSize,
                                                          const Vector4f& sceneIntrinsics,
                                                          const Matrix4f& approxInvPose, const Matrix4f& scenePose,
                                                          const float approxScaleFactor,
                                                          float distThresh)
{
	float depthValue = depth[PixelCoordsToIndex(x, y, viewImageSize)];
	if (depthValue <= 1e-8f) return false; //check if valid -- != 0.0f
	Vector4f depthPoint_view = Vector4f(reprojectImagePoint(x, y, depthValue, invertProjectionParams(viewIntrinsics)), 1);

	Vector3f depthPoint_rotated = (approxInvPose * Vector4f(depthPoint_view.toVector3(), 0)).toVector3();
	Vector3f depthPoint_world = approxScaleFactor * depthPoint_rotated + approxInvPose.getColumn(3).toVector3();

	// project into previous rendered image (scene)
	Vector4f depthPoint_scene = scenePose * Vector4f(depthPoint_world, 1);
	if (depthPoint_scene.z <= 0.0f) return false;
	Vector2f depthPoint2DCoords = project(depthPoint_scene.toVector3(), sceneIntrinsics);

	if (!((depthPoint2DCoords.x >= 0.0f) && (depthPoint2DCoords.x <= sceneImageSize.x - 2) &&
	      (depthPoint2DCoords.y >= 0.0f) &&
	      (depthPoint2DCoords.y <= sceneImageSize.y - 2)))
		return false;

	Vector2f sceneCoords = depthPoint2DCoords;

	// Corresponding point in rendered image (scene)
	Vector4f scenePoint_world = interpolateBilinear_withHoles(pointsMap, sceneCoords, sceneImageSize);
	if (scenePoint_world.w < 0.0f) return false;

	Vector3f ptDiff = scenePoint_world.toVector3() - depthPoint_world;
	float dist = ORUtils::dot(ptDiff, ptDiff);
	if (dist > distThresh or dist != dist) return false;

	Vector3f sceneNormal = interpolateBilinear_withHoles(normalsMap, sceneCoords, sceneImageSize).toVector3();


	f = dist;
	// multiplicative scale diff
//	j = Vector4f(-sceneNormal, -ORUtils::dot(approxScaleFactor * depthPoint_rotated, sceneNormal));
//	r = ORUtils::dot((scenePoint_world - approxInvPose.getColumn(3)).toVector3(), sceneNormal);

	// additive scale diff
	j = Vector4f(-sceneNormal, -ORUtils::dot(depthPoint_rotated, sceneNormal));
	r = ORUtils::dot(scenePoint_world.toVector3() - depthPoint_world, sceneNormal);

	// only translation (equal to default ICP)
//	j = Vector4f(-sceneNormal, 0);
//	r = ORUtils::dot(scenePoint_world.toVector3() - depthPoint_world, sceneNormal);

	return true;
}

template<bool shortIteration, bool rotationOnly>
_CPU_AND_GPU_CODE_ inline bool computePerPointGH_Depth(float* localNabla, float* localHessian, float& localF,
                                                       const int& x, const int& y,
                                                       const float* depth, const Vector2i& viewImageSize,
                                                       const Vector4f& viewIntrinsics, const Vector2i& sceneImageSize,
                                                       const Vector4f& sceneIntrinsics, const Matrix4f& approxInvPose,
                                                       const Matrix4f& scenePose, const Vector4f* pointsMap,
                                                       const Vector4f* normalsMap, float scaleFactor, float distThresh)
{
	const int noPara = shortIteration ? 3 : 6;
	float A[noPara];
	float b;

	bool ret = computePerPointGH_Depth_Ab<shortIteration, rotationOnly>(A, b, x, y, depth, viewImageSize, viewIntrinsics,
	                                                                    sceneImageSize, sceneIntrinsics, approxInvPose,
	                                                                    scenePose, pointsMap, normalsMap, scaleFactor,
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
