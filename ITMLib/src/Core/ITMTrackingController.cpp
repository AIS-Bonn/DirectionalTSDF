//
// Created by Malte Splietker on 28.10.21.
//

#include <ITMLib/Core/ITMTrackingController.h>
#include <Utils/ITMTimer.h>

namespace ITMLib {

void ITMTrackingController::Track(ITMTrackingState* trackingState, const ITMView* view)
{
	timeStats.Reset();
	ITMTimer timer;
	timer.Tick();
	tracker->TrackCamera(trackingState, view);
	timeStats.tracking = timer.Tock();
}

void ITMTrackingController::Prepare(ITMTrackingState* trackingState, const Scene* scene, const ITMView* view,
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
		trackingState->pose_pointCloud->SetFrom(trackingState->pose_d);
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

ITMTrackingController::ITMTrackingController(ITMTracker* tracker, const std::shared_ptr<const ITMLibSettings>& settings)
{
this->tracker = tracker;
this->settings = settings;
}

} // namespace ITMLib
