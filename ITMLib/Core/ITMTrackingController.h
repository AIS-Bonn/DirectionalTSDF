// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <stdexcept>

#include "ITMLib/Engines/Visualisation/Interface/ITMVisualisationEngine.h"
#include "ITMLib/Trackers/Interface/ITMTracker.h"
#include "ITMLib/Objects/Stats/ITMTrackingTimeStats.h"
#include "ITMLib/Utils/ITMLibSettings.h"
#include "ITMLib/Utils/ITMTimer.h"

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
	void Track(ITMTrackingState* trackingState, const ITMView* view)
	{
		timeStats.Reset();
		ITMTimer timer;
		timer.Tick();
		tracker->TrackCamera(trackingState, view);
		timeStats.tracking = timer.Tock();
	}

	template<typename TVoxel>
	void Prepare(ITMTrackingState* trackingState, const ITMScene<TVoxel>* scene, const ITMView* view,
	             ITMVisualisationEngine* visualisationEngine, ITMRenderState* renderState)
	{
		ITMTimer timer;
		timer.Tick();
		if (!tracker->requiresPointCloudRendering())
			return;

		//render for tracking
		bool requiresColourRendering = tracker->requiresColourRendering();
		bool requiresFullRendering = trackingState->TrackerFarFromPointCloud() || !settings->useApproximateRaycast;

		if (requiresColourRendering)
		{
			ORUtils::SE3Pose pose_rgb(view->calib.trafo_rgb_to_depth.calib_inv * trackingState->pose_d->GetM());
			visualisationEngine->CreateExpectedDepths(scene, &pose_rgb, &(view->calib.intrinsics_rgb), renderState);
			visualisationEngine->CreatePointCloud(scene, view, trackingState, renderState, settings->skipPoints);
			trackingState->age_pointCloud = 0;
		} else
		{
			visualisationEngine->CreateExpectedDepths(scene, trackingState->pose_d, &(view->calib.intrinsics_d), renderState);

			if (requiresFullRendering)
			{
				visualisationEngine->CreateICPMaps(scene, view, trackingState, renderState);
				trackingState->pose_pointCloud->SetFrom(trackingState->pose_d);
				if (trackingState->age_pointCloud == -1) trackingState->age_pointCloud = -2;
				else trackingState->age_pointCloud = 0;
			} else
			{
				visualisationEngine->ForwardRender(scene, view, trackingState, renderState);
				trackingState->age_pointCloud++;
			}
		}
		float t = timer.Tock();
		timeStats.rendering += t - visualisationEngine->renderingTSDFTime;
		timeStats.renderingTSDF = visualisationEngine->renderingTSDFTime;
	}

	ITMTrackingController(ITMTracker* tracker, const std::shared_ptr<const ITMLibSettings>& settings)
	{
		this->tracker = tracker;
		this->settings = settings;
	}

	const Vector2i& GetTrackedImageSize(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d) const
	{
		return tracker->requiresColourRendering() ? imgSize_rgb : imgSize_d;
	}

	// Suppress the default copy constructor and assignment operator
	ITMTrackingController(const ITMTrackingController&);

	ITMTrackingController& operator=(const ITMTrackingController&);

	const ITMTrackingTimeStats& GetTimeStats() const
	{
		return timeStats;
	}
};
}
