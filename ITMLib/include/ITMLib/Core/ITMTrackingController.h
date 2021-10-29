// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <stdexcept>

#include <ITMLib/Engines//ITMVisualisationEngine.h>
#include <ITMLib/Trackers/ITMTracker.h>
#include <ITMLib/Objects/Stats/ITMTrackingTimeStats.h>
#include <ITMLib/Utils/ITMLibSettings.h>

namespace ITMLib
{
/** \brief
*/
class ITMTrackingController
{
private:
	std::shared_ptr<const ITMLibSettings> settings;
	ITMTracker* tracker;
	ITMTrackingTimeStats timeStats;

public:
	ITMTrackingController(ITMTracker* tracker, const std::shared_ptr<const ITMLibSettings>& settings);

	// Suppress the default copy constructor and assignment operator
	ITMTrackingController(const ITMTrackingController&) = delete;

	ITMTrackingController& operator=(const ITMTrackingController&) = delete;

	void Track(ITMTrackingState* trackingState, const ITMView* view);

	void Prepare(ITMTrackingState* trackingState, const Scene* scene, const ITMView* view,
	             ITMVisualisationEngine* visualisationEngine, ITMRenderState* renderState);

	[[nodiscard]] const Vector2i& GetTrackedImageSize(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d) const
	{
		return tracker->requiresColourRendering() ? imgSize_rgb : imgSize_d;
	}

	[[nodiscard]] const ITMTrackingTimeStats& GetTimeStats() const
	{
		return timeStats;
	}
};
}
