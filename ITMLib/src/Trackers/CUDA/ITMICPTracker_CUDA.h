// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <ITMLib/Trackers/ITMICPTracker.h>

namespace ITMLib
{
class ITMICPTracker_CUDA : public ITMICPTracker
{
public:
	struct AccuCell;

private:
	AccuCell* accumulator_device = nullptr;

protected:
	int ComputeGandH_Depth(float& f, float* nabla, float* hessian, Matrix4f approxInvPose, float approxScaleFactor) override;

	int ComputeGandH_RGB(float& f, float* nabla, float* hessian, Matrix4f approxInvPose, float approxScaleFactor) override;

	size_t ComputeTransScale(float& f, Eigen::Matrix<EigenT, 4, 4>& H, Eigen::Matrix<EigenT, 4, 1>& g, const Matrix4f& approxInvPose, float approxScaleFactor) override;

	size_t ComputeSahillioglu(float& f, Eigen::Matrix<EigenT, 4, 4>& A, Eigen::Matrix<EigenT, 4, 1>& b, const Matrix4f& approxInvPose, float approxScaleFactor) override;

	void ComputeDepthPointAndIntensity(ITMFloat4Image* points_out,
	                                   ITMFloatImage* intensity_out,
	                                   const ITMFloatImage* intensity_in,
	                                   const ITMFloatImage* depth_in,
	                                   const Vector4f& intrinsics_depth,
	                                   const Vector4f& intrinsics_rgb,
	                                   const Matrix4f& scenePose) override;

public:
	ITMICPTracker_CUDA(Vector2i imgSize_d, Vector2i imgSize_rgb, const Parameters& parameters,
	                   const ITMLowLevelEngine* lowLevelEngine);

	~ITMICPTracker_CUDA(void);
};
}
