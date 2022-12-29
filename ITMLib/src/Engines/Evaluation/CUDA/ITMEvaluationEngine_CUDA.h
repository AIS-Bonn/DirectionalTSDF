//
// Created by Malte Splietker on 28.12.22.
//

#pragma once

#include <ITMLib/Engines/ITMEvaluationEngine.h>
#include <ITMLib/Utils/ITMImageTypes.h>

namespace ITMLib
{
class ITMEvaluationEngine_CUDA : public ITMEvaluationEngine
{
public:
	ITMRenderError ComputeICPError(
		const ITMFloatImage& depth,
		const ORUtils::Image<ORUtils::Vector4<float>>& points,
		const ORUtils::Image<ORUtils::Vector4<float>>& normals,
		const Matrix4f& depthImageInvPose,
		const Matrix4f& sceneRenderingPose,
		ITMIntrinsics& intrinsics_d
	) const override;

	ITMRenderError ComputePhotometricError(
		const ITMUChar4Image& image,
		const ITMUChar4Image& render,
		const ITMFloat4Image& points,
		const Matrix4f& depthImageInvPose,
		const Matrix4f& sceneRenderingPose,
		ITMIntrinsics& intrinsics_rgb
	) const override;

	ITMEvaluationEngine_CUDA();
};
}
