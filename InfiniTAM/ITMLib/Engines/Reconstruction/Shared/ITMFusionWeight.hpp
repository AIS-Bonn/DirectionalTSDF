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
inline float weightNormal(Vector4f normalCamera)
{
	return -normalCamera.z;
}

_CPU_AND_GPU_CODE_
inline float depthWeight(float depth, const Vector4f& normalCamera, float directionWeight,
                         const ITMSceneParams& sceneParams)
{
	return weightDepth(depth, sceneParams) * weightNormal(normalCamera) * directionWeight;
}

_CPU_AND_GPU_CODE_
inline float colorWeight(float normalizedDepth, const Vector4f& normalCamera, float directionWeight,
                         const ITMSceneParams& sceneParams)
{
	return depthWeight(normalizedDepth, normalCamera, directionWeight, sceneParams);
}

} // namespace ITMLib
