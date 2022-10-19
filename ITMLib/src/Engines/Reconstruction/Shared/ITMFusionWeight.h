//
// Created by Malte Splietker on 20.05.19.
//

#pragma once

#include <ITMLib/Objects/Scene/ITMDirectional.h>
#include <ITMLib/Utils/ITMSceneParams.h>

namespace ITMLib
{

_CPU_AND_GPU_CODE_
inline float depthNoiseSigma(float z, float theta)
{
	return 0.0012f + 0.019f * (z - 0.4f) * (z - 0.4f) +
	       0.0001f / std::sqrt(z) * theta * theta / ((0.5f * float(M_PI) - theta) * (0.5f * float(M_PI) - theta));
}

_CPU_AND_GPU_CODE_
inline float weightDepth(float depth, const ITMSceneParams& sceneParams)
{
	// Newcombe2011 (KinectFusion)
//	return 1 / depth;

	// Nguyen2012
//	return sigma(sceneParams.viewFrustum_min, 0) / depthNoiseSigma(depth, 0)
//         * (sceneParams.viewFrustum_min * sceneParams.viewFrustum_min) / (depth * depth);

	// Normalized, s.t. weight of minimum distance is 1
	return (sceneParams.viewFrustum_min * sceneParams.viewFrustum_min) / (depth * depth);

	// Not normalized, but independent of min/max distance
//	return MIN(1 / (depth * depth), 1);
}

_CPU_AND_GPU_CODE_
inline float weightNormal(const Vector3f& normalCamera, const Vector3f& viewRay)
{
	return dot(normalCamera, -viewRay);
}

_CPU_AND_GPU_CODE_
inline float weightVisibility(float distance, const ITMSceneParams& sceneParams)
{
	// Unity
//	if (distance >= -1)
//		return 1;
//	return 0;

	// KinectFusion
//	if (distance > 0)
//		return 1;
//	return MAX(0, (1 + distance) / sceneParams.mu);

	// CM3D CopyMe3D: Scanning and Printing Persons in 3D
	if (distance > 0)
		return 1;
	return MAX(0.1f, std::exp(-(distance * distance) / (sceneParams.mu * sceneParams.mu)));
}

_CPU_AND_GPU_CODE_
inline float combinedWeight(float depth, float distance, const Vector3f& normalCamera, const Vector3f& viewRay,
                            const ITMSceneParams& sceneParams)
{
	return weightVisibility(distance, sceneParams) * weightDepth(depth, sceneParams) *
	       weightNormal(normalCamera, viewRay);
}

} // namespace ITMLib
