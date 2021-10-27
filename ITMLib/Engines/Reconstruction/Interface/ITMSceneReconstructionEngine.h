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
class IITMSceneReconstructionEngine
{
public:
	/** Clear and reset a scene to set up a new empty
			one.
	*/
	virtual void ResetScene(Scene* scene) = 0;

	/** Given a view with a new depth image, compute the
			visible blocks, allocate in the TSDF so that the
			new image data can be integrated.
	*/
	virtual void AllocateSceneFromDepth(Scene* scene, const ITMView* view, const ITMTrackingState* trackingState) = 0;

	/** Given a scene, pose and projParams, compute the
	visible subset of the scene and store it in an
	appropriate visualisation state object, created
	previously using allocateInternalState().
	*/
	virtual void FindVisibleBlocks(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                               ITMRenderState* renderState) = 0;

	/** Update the voxel blocks by integrating depth and
			possibly colour information from the given view.
	*/
	virtual void IntegrateIntoScene(Scene* scene, const ITMView* view, const ITMTrackingState* trackingState) = 0;

	IITMSceneReconstructionEngine() = default;

	virtual ~IITMSceneReconstructionEngine() = default;

	virtual ITMReconstructionTimeStats& GetTimeStats() = 0;

	[[nodiscard]] virtual const ITMReconstructionTimeStats& GetTimeStats() const = 0;
};

template<typename TIndex>
class ITMSceneReconstructionEngine : public IITMSceneReconstructionEngine
{
public:
	explicit ITMSceneReconstructionEngine(std::shared_ptr<const ITMLibSettings> settings, MemoryDeviceType memoryDevice)
		: settings(std::move(settings))
	{
		allocationFusionBlocksList = new ORUtils::MemoryBlock<TIndex>(10000, memoryDevice);
	}

	void IntegrateIntoScene(Scene* scene, const ITMView* view, const ITMTrackingState* trackingState) override
	{
		if (this->settings->fusionParams.fusionMode == FusionMode::FUSIONMODE_VOXEL_PROJECTION)
		{
			IntegrateIntoSceneVoxelProjection(scene, view, trackingState);
		} else
		{
			IntegrateIntoSceneRayCasting(scene, view, trackingState);
		}
	}

	ITMReconstructionTimeStats& GetTimeStats() override
	{
		return timeStats;
	}

	[[nodiscard]] const ITMReconstructionTimeStats& GetTimeStats() const override
	{
		return timeStats;
	}

	ITMSceneReconstructionEngine() = default;

	~ITMSceneReconstructionEngine() override = default;

protected:
	static const bool directional = std::is_same<TIndex, ITMIndexDirectional>::value;

	std::shared_ptr<const ITMLibSettings> settings;

	/** List of blocks used during allocation and fusion process.
	 * During fusion the list may contain additional blocks. */
	ORUtils::MemoryBlock<TIndex>* allocationFusionBlocksList{};

	ITMReconstructionTimeStats timeStats;

	size_t noAllocationBlocks = 0;
	size_t noFusionBlocks = 0;

	virtual void IntegrateIntoSceneVoxelProjection(Scene* scene, const ITMView* view,
	                                               const ITMTrackingState* trackingState) = 0;

	virtual void IntegrateIntoSceneRayCasting(Scene* scene, const ITMView* view,
	                                          const ITMTrackingState* trackingState) = 0;
};
}
