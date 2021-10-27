// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMICPTracker.h"

namespace ITMLib
{
class ITMICPTracker_CPU : public ITMICPTracker
{
protected:
	int ComputeGandH_Depth(float& f, float* nabla, float* hessian, Matrix4f approxInvPose) override;

	int ComputeGandH_RGB(float& f, float* nabla, float* hessian, Matrix4f approxInvPose) override;

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
