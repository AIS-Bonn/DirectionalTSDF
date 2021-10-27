// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMVisualisationEngine.h"

namespace ITMLib
{
class ITMVisualisationEngine_CPU : public ITMVisualisationEngine
{
public:
	explicit ITMVisualisationEngine_CPU(const std::shared_ptr<const ITMLibSettings>& settings);

	~ITMVisualisationEngine_CPU() override = default;

	void FindSurface(const Scene* scene, const ORUtils::SE3Pose* pose,
	                 const ITMIntrinsics* intrinsics, const ITMRenderState* renderState) const override;

	void RenderImage(const Scene* scene,
	                 const ORUtils::SE3Pose* pose,
	                 const ITMIntrinsics* intrinsics,
	                 const ITMRenderState* renderState,
	                 ITMUChar4Image* outputImage,
	                 IITMVisualisationEngine::RenderImageType type,
	                 IITMVisualisationEngine::RenderRaycastSelection raycastType) const override;

	void CreatePointCloud(const Scene* scene, const ITMView* view,
	                      ITMTrackingState* trackingState, ITMRenderState* renderState, bool skipPoints) const override;

	void CreateICPMaps(const Scene* scene, const ITMView* view,
	                   ITMTrackingState* trackingState, ITMRenderState* renderState) const override;

	void ForwardRender(const Scene* scene, const ITMView* view,
	                   ITMTrackingState* trackingState, ITMRenderState* renderState) const override;

	void RenderTrackingError(ITMUChar4Image* outRendering, const ITMTrackingState* trackingState,
	                         const ITMView* view) const override;

	ITMRenderState* CreateRenderState(const Scene* scene, const Vector2i& imgSize) const override;

	void CreateExpectedDepths(const Scene* scene, const ORUtils::SE3Pose* pose,
	                          const ITMIntrinsics* intrinsics, ITMRenderState* renderState) override;

protected:
	void GenericRaycast(const Scene* scene, const Vector2i& imgSize, const Matrix4f& invM,
	                    const Vector4f& projParams, const ITMRenderState* renderState, bool updateVisibleList) const;

	void ComputeRenderingTSDFImpl(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                              ITMRenderState* renderState) override;
};

}
