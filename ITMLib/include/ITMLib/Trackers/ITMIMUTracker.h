// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <ITMLib/Trackers/ITMTracker.h>
#include <ITMLib/Engines/ITMLowLevelEngine.h>
#include <ITMLib/Objects/Misc/ITMIMUCalibrator.h>
#include <ITMLib/Objects/Misc/ITMIMUMeasurement.h>

namespace ITMLib
{
class ITMIMUTracker : public ITMTracker
{
private:
	ITMIMUCalibrator* calibrator;

public:
	void TrackCamera(ITMTrackingState* trackingState, const ITMView* view);

	bool requiresColourRendering() const
	{ return false; }

	bool requiresDepthReliability() const
	{ return false; }

	bool requiresPointCloudRendering() const
	{ return false; }

	ITMIMUTracker(ITMIMUCalibrator* calibrator);

	virtual ~ITMIMUTracker(void);
};
}
