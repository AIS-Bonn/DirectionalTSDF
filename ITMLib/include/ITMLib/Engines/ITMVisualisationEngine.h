#include <utility>

#include <memory>
#include <utility>

// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <ITMLib/Objects/RenderStates/ITMRenderState.h>
#include <ITMLib/Objects/Scene/ITMScene.h>
#include <ITMLib/Objects/Tracking/ITMTrackingState.h>
#include <ITMLib/Objects/Views/ITMView.h>
#include <ITMLib/Utils/ITMLibSettings.h>

namespace ITMLib
{
class IITMVisualisationEngine
{
public:
	enum RenderImageType
	{
		RENDER_DEPTH_SDFNORMAL,
		RENDER_DEPTH_IMAGENORMAL,
		RENDER_COLOUR,
		RENDER_NORMAL_SDFNORMAL,
		RENDER_NORMAL_IMAGENORMAL,
		RENDER_CONFIDENCE_SDFNORMAL,
		RENDER_CONFIDENCE_IMAGENORMAL,
		RENDER_DEPTH_COLOUR, // rainbow-style depth rendering
	};

	enum RenderRaycastSelection
	{
		RENDER_FROM_NEW_RAYCAST,
		RENDER_FROM_OLD_RAYCAST,
		RENDER_FROM_OLD_FORWARDPROJ
	};

	virtual ~IITMVisualisationEngine() = default;

	static void DepthToUchar4(ITMUChar4Image* dst, const ITMFloatImage* src);

	static void NormalToUchar4(ITMUChar4Image* dst, const ITMFloat4Image* src);

	static void WeightToUchar4(ITMUChar4Image* dst, const ITMFloatImage* src);
};


/** \brief
	Interface to engines helping with the visualisation of
	the results from the rest of the library.

	This is also used internally to get depth estimates for the
	raycasting done for the trackers. The basic idea there is
	to project down a scene of 8x8x8 voxel
	blocks and look at the bounding boxes. The projection
	provides an idea of the possible depth range for each pixel
	in an image, which can be used to speed up raycasting
	operations.
	*/
class ITMVisualisationEngine : public IITMVisualisationEngine
{
public:
	explicit ITMVisualisationEngine(std::shared_ptr<const ITMLibSettings> settings)
		: settings(std::move(settings))
	{}

	virtual ~ITMVisualisationEngine()
	{
		delete renderingTSDF;
	}


	/** Creates a render state, containing rendering info
	for the scene.
	*/
	virtual ITMRenderState*
	CreateRenderState(const Scene* scene, const Vector2i& imgSize) const = 0;

	/** Given scene, pose and projParams, create an estimate
	of the minimum and maximum depths at each pixel of
	an image.
	*/
	virtual void CreateExpectedDepths(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                                  ITMRenderState* renderState) = 0;

	/** This will render an image using raycasting. */
	virtual void RenderImage(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                         const ITMRenderState* renderState, ITMUChar4Image* outputImage,
	                         RenderImageType type = RENDER_DEPTH_SDFNORMAL,
	                         RenderRaycastSelection raycastType = RENDER_FROM_NEW_RAYCAST) const = 0;

	/** Finds the scene surface using raycasting. */
	virtual void FindSurface(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                         const ITMRenderState* renderState) const = 0;

	/** Create a point cloud as required by the
	ITMLib::Engine::ITMColorTracker classes.
	*/
	virtual void CreatePointCloud(const Scene* scene, ITMIntrinsics intrinsics, const ORUtils::SE3Pose* pose,
	                              ITMPointCloud* pointCloud, ITMRenderState* renderState, bool skipPoints) const = 0;

	/** Create an image of reference points and normals as
	required by the ITMLib::Engine::ITMICPTracker classes.
	*/
	virtual void CreateICPMaps(const Scene* scene, ITMIntrinsics intrinsics, const ORUtils::SE3Pose* pose,
	                           ITMPointCloud* pointCloud, ITMRenderState* renderState) const = 0;

	/** Create an image of reference points and normals as
	required by the ITMLib::Engine::ITMICPTracker classes.

	Incrementally previous raycast result.
	*/
	virtual void ForwardRender(const Scene* scene, const ITMView* view, ITMTrackingState* trackingState,
	                           ITMRenderState* renderState) const = 0;

	virtual void RenderTrackingError(ITMUChar4Image* outRendering, const ITMTrackingState* trackingState,
	                                 const ITMView* view) const = 0;

	float renderingTSDFTime = 0;

protected:
	virtual void
	ComputeRenderingTSDFImpl(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                         ITMRenderState* renderState) = 0;

	void ComputeRenderingTSDF(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                          ITMRenderState* renderState);

	/**
	 * Returns current renderingTSDF or scene->tsdf, depending on whether directional is enabled or not
	 */
	TSDF<ITMIndex, ITMVoxel>* GetRenderingTSDF(const Scene* scene) const
	{
		if (settings->Directional())
			return renderingTSDF;
		return scene->tsdf;
	}

	TSDF<ITMIndex, ITMVoxel>* renderingTSDF = nullptr;

	std::shared_ptr<const ITMLibSettings> settings = nullptr;

	int frameCounter = 0;
	int lastTSDFCombineFrameCounter = 0;
	ORUtils::SE3Pose lastTSDFCombinePose;
};
}