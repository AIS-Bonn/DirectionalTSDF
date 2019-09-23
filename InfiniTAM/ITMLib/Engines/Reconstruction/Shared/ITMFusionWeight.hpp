//
// Created by Malte Splietker on 20.05.19.
//

#pragma once

namespace ITMLib
{

_CPU_AND_GPU_CODE_
inline float weightDepth(float depth, const ITMSceneParams& sceneParams)
{
	float normalizedDepth =
		(depth - sceneParams.viewFrustum_min) / (sceneParams.viewFrustum_max - sceneParams.viewFrustum_min);
	return 1.0f - normalizedDepth * normalizedDepth;
}

_CPU_AND_GPU_CODE_
inline float weightNormal(const Vector3f &normalCamera, const Vector3f &viewRay)
{
	return dot(normalCamera, -viewRay);
}

_CPU_AND_GPU_CODE_
inline float depthWeight(float depth, const Vector3f& normalCamera, const Vector3f &viewRay,
	float directionWeight, const ITMSceneParams& sceneParams)
{
	return weightDepth(depth, sceneParams) * weightNormal(normalCamera, viewRay) * directionWeight;
}

_CPU_AND_GPU_CODE_
inline float colorWeight(float depth, const Vector3f& normalCamera, const Vector3f &viewRay,
                         float directionWeight, const ITMSceneParams& sceneParams)
{
	return depthWeight(depth, normalCamera, viewRay, directionWeight, sceneParams);
}

} // namespace ITMLib
