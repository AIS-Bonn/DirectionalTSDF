// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMLowLevelEngine.h"

namespace ITMLib
{
class ITMLowLevelEngine_CUDA : public ITMLowLevelEngine
{
private:
	int* counterTempData_device{}, * counterTempData_host{};

public:
	void CopyImage(ITMUChar4Image* image_out, const ITMUChar4Image* image_in) const override;

	void CopyImage(ITMFloatImage* image_out, const ITMFloatImage* image_in) const override;

	void CopyImage(ITMFloat4Image* image_out, const ITMFloat4Image* image_in) const override;

	void ConvertColourToIntensity(ITMFloatImage* image_out, const ITMUChar4Image* image_in) const override;

	void FilterIntensity(ITMFloatImage* image_out, const ITMFloatImage* image_in) const override;

	void FilterSubsample(ITMUChar4Image* image_out, const ITMUChar4Image* image_in) const override;

	void FilterSubsample(ITMFloatImage* image_out, const ITMFloatImage* image_in) const override;

	void FilterSubsampleWithHoles(ITMFloatImage* image_out, const ITMFloatImage* image_in) const override;

	void FilterSubsampleWithHoles(ITMFloat4Image* image_out, const ITMFloat4Image* image_in) const override;

	void GradientX(ITMShort4Image* grad_out, const ITMUChar4Image* image_in) const override;

	void GradientY(ITMShort4Image* grad_out, const ITMUChar4Image* image_in) const override;

	void GradientXY(ITMFloat2Image* grad_out, const ITMFloatImage* image_in) const override;

	int CountValidDepths(const ITMFloatImage* image_in) const override;

	ITMLowLevelEngine_CUDA();

	~ITMLowLevelEngine_CUDA() override;
};
}
