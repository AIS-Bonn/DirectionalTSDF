// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <ITMLib/Trackers/ITMICPTracker.h>

namespace ITMLib
{
class ITMICPTracker_CPU : public ITMICPTracker
{
protected:
	int ComputeGandH_Depth(float& f, float* nabla, float* hessian, const Matrix4f& deltaT) override;

	int ComputeGandH_RGB(float& f, float* nabla, float* hessian, const Matrix4f& deltaT) override;

	void RenderRGBError(ITMUChar4Image* image_out, const Matrix4f& deltaT) override {};

	size_t ComputeGandHSim3_Depth(float& f, Eigen::Matrix<EigenT, 7, 7>& H, Eigen::Matrix<EigenT, 7, 1>& g, const Matrix4f& approxInvPose) override;

	size_t ComputeGandHSim3_RGB(float& f, Eigen::Matrix<EigenT, 7, 7>& H, Eigen::Matrix<EigenT, 7, 1>& g, const Matrix4f& approxInvPose) override;

	void ComputeDepthPointAndIntensity(ITMFloat4Image* points_out,
	                                   ITMFloatImage* intensity_out,
	                                   const ITMFloatImage* intensity_in,
	                                   const ITMFloatImage* depth_in,
	                                   const Vector4f& intrinsics_depth,
	                                   const Vector4f& intrinsics_rgb,
	                                   const Matrix4f& scenePose) override;

public:
	ITMICPTracker_CPU(Vector2i imgSize_d, Vector2i imgSize_rgb, const Parameters& parameters,
	                  const ITMLowLevelEngine* lowLevelEngine);

	~ITMICPTracker_CPU(void);
};
}
