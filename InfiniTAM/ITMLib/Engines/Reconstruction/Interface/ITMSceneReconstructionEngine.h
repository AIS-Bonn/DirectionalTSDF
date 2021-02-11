// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <memory>
#include <ITMLib/ITMLibDefines.h>

#include "ITMLib/Objects/RenderStates/ITMRenderState.h"
#include "ITMLib/Objects/Scene/ITMScene.h"
#include "ITMLib/Objects/Tracking/ITMTrackingState.h"
#include "ITMLib/Objects/Views/ITMView.h"
#include "ITMLib/Utils/ITMLibSettings.h"
#include "ITMLib/Objects/Stats/ITMReconstructionTimeStats.h"
#include "ITMLib/Objects/Scene/ITMSummingVoxel.h"

namespace ITMLib
{

/** \brief
		Interface to engines implementing the main KinectFusion
		depth integration process.

		These classes basically manage
		an ITMLib::Objects::ITMScene and fuse new image information
		into them.
*/
class ITMSceneReconstructionEngine
{
public:
	explicit ITMSceneReconstructionEngine(std::shared_ptr<const ITMLibSettings> settings)
		: settings(std::move(settings))
	{}

	/** Clear and reset a scene to set up a new empty
			one.
	*/
	virtual void ResetScene(Scene* scene) = 0;

	/** Given a view with a new depth image, compute the
			visible blocks, allocate them and update the hash
			table so that the new image data can be integrated.
	*/
	virtual void AllocateSceneFromDepth(Scene* scene, const ITMView* view, const ITMTrackingState* trackingState,
	                                    const ITMRenderState* renderState, bool onlyUpdateVisibleList = false,
	                                    bool resetVisibleList = false) = 0;

	/** Update the voxel blocks by integrating depth and
			possibly colour information from the given view.
	*/
	void IntegrateIntoScene(Scene* scene, const ITMView* view,
	                        const ITMTrackingState* trackingState, const ITMRenderState* renderState)
	{
		if (this->settings->fusionParams.fusionMode == FusionMode::FUSIONMODE_VOXEL_PROJECTION)
		{
			IntegrateIntoSceneVoxelProjection(scene, view, trackingState, renderState);
		} else
		{
			IntegrateIntoSceneRayCasting(scene, view, trackingState, renderState);
		}
	}

	ITMReconstructionTimeStats& GetTimeStats()
	{
		return timeStats;
	}

	const ITMReconstructionTimeStats& GetTimeStats() const
	{
		return timeStats;
	}

	ITMSceneReconstructionEngine(void)
	{}

	virtual ~ITMSceneReconstructionEngine(void)
	{}

protected:
	std::shared_ptr<const ITMLibSettings> settings;

	ITMReconstructionTimeStats timeStats;

	virtual void IntegrateIntoSceneVoxelProjection(Scene* scene, const ITMView* view,
	                                               const ITMTrackingState* trackingState,
	                                               const ITMRenderState* renderState) = 0;

	virtual void IntegrateIntoSceneRayCasting(Scene* scene, const ITMView* view,
	                                          const ITMTrackingState* trackingState,
	                                          const ITMRenderState* renderState) = 0;
};
}
