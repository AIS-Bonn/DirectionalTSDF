// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Camera/ITMCalibIO.h"
#include "../../Utils/ITMImageTypes.h"

namespace ITMLib
{
/** \brief
		Represents a single "view", i.e. RGB and depth images along
		with all intrinsic and relative calibration information
*/
class ITMView
{
public:
	/// Intrinsic calibration information for the view.
	const ITMRGBDCalib calib;

	/// RGB colour image for the current frame.
	ITMUChar4Image* rgb = nullptr;

	/// Float valued depth image, if available according to @ref inputImageType.
	ITMFloatImage* depth = nullptr;

	/// surface normal of depth image
	ITMFloat4Image* depthNormal = nullptr;

	ITMView(const ITMRGBDCalib& calibration, Vector2i imgSize_rgb, Vector2i imgSize_d, bool useGPU)
		: calib(calibration)
	{
		this->rgb = new ITMUChar4Image(imgSize_rgb, true, useGPU);
		this->depth = new ITMFloatImage(imgSize_d, true, useGPU);
		this->depthNormal = new ITMFloat4Image(imgSize_d, true, useGPU);
	}

	virtual ~ITMView(void)
	{
		delete rgb;
		delete depth;
		delete depthNormal;
	}

	// Suppress the default copy constructor and assignment operator
	ITMView(const ITMView&);

	ITMView& operator=(const ITMView&);
};
}
