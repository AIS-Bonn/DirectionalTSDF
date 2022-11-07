// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <memory>
#include <ITMLib/Objects/Scene/TSDF.h>
#include <ITMLib/Engines/ITMVisualisationEngine.h>
#include <stdgpu/unordered_map_fwd>

namespace ITMLib
{
struct RenderingBlock;

class ITMVisualisationEngine_CUDA : public ITMVisualisationEngine
{
public:
	explicit ITMVisualisationEngine_CUDA(std::shared_ptr<const ITMLibSettings> settings);

	~ITMVisualisationEngine_CUDA() override;

	void CreatePointCloud(const Scene* scene, ITMIntrinsics intrinsics, const ORUtils::SE3Pose* pose,
	                              ITMPointCloud* pointCloud, ITMRenderState* renderState, bool skipPoints) const override;

	void ForwardRender(const Scene* scene, const ITMView* view,
	                   ITMTrackingState* trackingState, ITMRenderState* renderState) const override;

	void RenderImage(const Scene* scene, const ORUtils::SE3Pose* pose,
	                 const ITMIntrinsics* intrinsics, const ITMRenderState* renderState,
	                 ITMUChar4Image* outputImage,
	                 IITMVisualisationEngine::RenderImageType type,
	                 IITMVisualisationEngine::RenderRaycastSelection raycastType) const override;

	void FindSurface(const Scene* scene, const ORUtils::SE3Pose* pose,
	                 const ITMIntrinsics* intrinsics, const ITMRenderState* renderState) const override;

	void CreateICPMaps(const Scene* scene, ITMIntrinsics intrinsics, const ORUtils::SE3Pose* pose,
	                   ITMPointCloud* pointCloud, ITMRenderState* renderState) const override;

	void RenderTrackingError(ITMUChar4Image* outRendering, const ITMTrackingState* trackingState,
	                         const ITMView* view) const override;

	ITMRenderState* CreateRenderState(const Scene* scene, const Vector2i& imgSize) const override;

	void CreateExpectedDepths(const Scene* scene, const ORUtils::SE3Pose* pose,
	                          const ITMIntrinsics* intrinsics, ITMRenderState* renderState) override;

protected:
	uint* noTotalPoints_device;

	void GenericRaycast(const Scene* scene, const Vector2i& imgSize, const Matrix4f& invM,
	                    const Vector4f& projParams, const ITMRenderState* renderState, bool updateVisibleList) const;

	void ComputeRenderingTSDFImpl(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                              ITMRenderState* renderState) override;

private:
	RenderingBlock* renderingBlockList_device;
	uint* noTotalBlocks_device;
	int* noVisibleEntries_device;
};

}
