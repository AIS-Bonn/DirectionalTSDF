#include <utility>

#include <memory>
#include <utility>

// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../../Objects/RenderStates/ITMRenderState_VH.h"
#include "../../../Objects/Scene/ITMScene.h"
#include "../../../Objects/Tracking/ITMTrackingState.h"
#include "../../../Objects/Views/ITMView.h"

namespace ITMLib
{
class IITMVisualisationEngine
{
public:
	enum RenderImageType
	{
		RENDER_SHADED_GREYSCALE,
		RENDER_SHADED_GREYSCALE_IMAGENORMALS,
		RENDER_COLOUR_FROM_VOLUME,
		RENDER_COLOUR_FROM_SDFNORMAL,
		RENDER_COLOUR_FROM_IMAGENORMAL,
		RENDER_COLOUR_FROM_CONFIDENCE_SDFNORMAL,
		RENDER_COLOUR_FROM_CONFIDENCE_IMAGENORMAL,
		RENDER_COLOUR_FROM_DEPTH,
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

template<class ITMVoxelIndex>
struct IndexToRenderState
{
	typedef ITMRenderState type;
};
template<>
struct IndexToRenderState<ITMVoxelBlockHash>
{
	typedef ITMRenderState_VH type;
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
		: settings(std::move(settings)),
		renderIndex(nullptr), renderVBA(nullptr), renderVisibleEntryIDs(nullptr), renderNoVisibleEntries(0)
	{}

	virtual ~ITMVisualisationEngine() = default;

	/** Creates a render state, containing rendering info
	for the scene.
	*/
	virtual typename IndexToRenderState<ITMVoxelIndex>::type*
	CreateRenderState(const Scene* scene, const Vector2i& imgSize) const = 0;

	/** Given a scene, pose and intrinsics, compute the
	visible subset of the scene and store it in an
	appropriate visualisation state object, created
	previously using allocateInternalState().
	*/
	virtual void FindVisibleBlocks(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                               ITMRenderState* renderState) const = 0;

	virtual void ComputeRenderingTSDF(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                                  ITMRenderState* renderState) = 0;

	/** Given a render state, Count the number of visible blocks
	with minBlockId <= blockID <= maxBlockId .
	*/
	virtual int CountVisibleBlocks(const Scene* scene, const ITMRenderState* renderState, int minBlockId = 0,
	                               int maxBlockId = SDF_LOCAL_BLOCK_NUM) const = 0;

	/** Given scene, pose and intrinsics, create an estimate
	of the minimum and maximum depths at each pixel of
	an image.
	*/
	virtual void CreateExpectedDepths(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                                  ITMRenderState* renderState) = 0;

	/** This will render an image using raycasting. */
	virtual void RenderImage(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                         const ITMRenderState* renderState, ITMUChar4Image* outputImage,
	                         RenderImageType type = RENDER_SHADED_GREYSCALE,
	                         RenderRaycastSelection raycastType = RENDER_FROM_NEW_RAYCAST) const = 0;

	/** Finds the scene surface using raycasting. */
	virtual void FindSurface(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                         const ITMRenderState* renderState) const = 0;

	/** Create a point cloud as required by the
	ITMLib::Engine::ITMColorTracker classes.
	*/
	virtual void CreatePointCloud(const Scene* scene, const ITMView* view, ITMTrackingState* trackingState,
	                              ITMRenderState* renderState, bool skipPoints) const = 0;

	/** Create an image of reference points and normals as
	required by the ITMLib::Engine::ITMDepthTracker classes.
	*/
	virtual void CreateICPMaps(const Scene* scene, const ITMView* view, ITMTrackingState* trackingState,
	                           ITMRenderState* renderState) const = 0;

	/** Create an image of reference points and normals as
	required by the ITMLib::Engine::ITMDepthTracker classes.

	Incrementally previous raycast result.
	*/
	virtual void ForwardRender(const Scene* scene, const ITMView* view, ITMTrackingState* trackingState,
	                           ITMRenderState* renderState) const = 0;

	virtual void RenderTrackingError(ITMUChar4Image* outRendering, const ITMTrackingState* trackingState,
	                                 const ITMView* view) const = 0;

	void SaveRenderTSDF(const std::string& outputDirectory)
	{
		if (not renderVBA or not renderIndex)
			return;
		renderVBA->SaveToDirectory(outputDirectory);
		renderIndex->SaveToDirectory(outputDirectory);
	}

protected:
	std::shared_ptr<const ITMLibSettings> settings;

	ITMVoxelBlockHash* renderIndex;
	ITMLocalVBA<ITMVoxel>* renderVBA;
	ORUtils::MemoryBlock<int> *renderVisibleEntryIDs;
	int renderNoVisibleEntries;

	/** True, if directional TSDF is combined to default TSDF for rendering
	 * @return
	 */
	bool CombineTSDFForRendering() const
	{
		return this->settings->fusionParams.tsdfMode == TSDFMode::TSDFMODE_DIRECTIONAL and DIRECTIONAL_RENDERING_MODE == 1;
	}
};
}
