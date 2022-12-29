//
// Created by Malte Splietker on 28.12.22.
//

#pragma once

#include <ITMLib/Objects/Camera/ITMIntrinsics.h>
#include <ITMLib/Objects/Stats/ITMRenderError.h>
#include <ITMLib/Utils/ITMImageTypes.h>

namespace ITMLib
{
/// Interface to low level image processing engines.
class ITMEvaluationEngine
{
public:
	virtual ITMRenderError ComputeICPError(
		const ITMFloatImage& depth,
		const ORUtils::Image<ORUtils::Vector4<float>>& points,
		const ORUtils::Image<ORUtils::Vector4<float>>& normals,
		const Matrix4f& depthImageInvPose,
		const Matrix4f& sceneRenderingPose,
		ITMIntrinsics& intrinsics_d
	) const = 0;

	virtual ITMRenderError ComputePhotometricError(
		const ITMUChar4Image& image,
		const ITMUChar4Image& render,
		const ITMFloat4Image& points,
		const Matrix4f& depthImageInvPose,
		const Matrix4f& sceneRenderingPose,
		ITMIntrinsics& intrinsics_rgb
	) const = 0;

	ITMEvaluationEngine(void)
	{}

	virtual ~ITMEvaluationEngine(void)
	{}
};
}
