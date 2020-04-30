//
// Created by Malte Splietker on 20.05.19.
//

#pragma once

namespace ITMLib
{

_CPU_AND_GPU_CODE_
inline float depthNoiseSigma(float z, float theta)
{
	return 0.0012 + 0.019 * (z - 0.4) * (z - 0.4) +
	       0.0001 / sqrt(z) * theta * theta / ((0.5 * M_PI - theta) * (0.5 * M_PI - theta));
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
inline float depthWeight(float depth, const Vector3f& normalCamera, const Vector3f& viewRay,
                         float directionWeight, const ITMSceneParams& sceneParams)
{
	return weightDepth(depth, sceneParams) * weightNormal(normalCamera, viewRay) * directionWeight;
}

_CPU_AND_GPU_CODE_
inline float colorWeight(float depth, const Vector3f& normalCamera, const Vector3f& viewRay,
                         float directionWeight, const ITMSceneParams& sceneParams)
{
	return depthWeight(depth, normalCamera, viewRay, directionWeight, sceneParams);
}

} // namespace ITMLib
