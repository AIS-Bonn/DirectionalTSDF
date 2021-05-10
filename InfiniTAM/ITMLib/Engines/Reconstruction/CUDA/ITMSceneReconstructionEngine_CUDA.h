// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMSceneReconstructionEngine.h"

namespace ITMLib
{
class SummingVoxelMap_CUDA;

class ITMSceneReconstructionEngine_CUDA : public ITMSceneReconstructionEngine
{
private:
	HashEntryAllocType *entriesAllocType_device;
	Vector4s *blockCoords_device;
	TSDFDirection *blockDirections_device;
	size_t noAllocationBlocks;
	size_t noFusionBlocks;

	SummingVoxelMap_CUDA* summingVoxelMap;

public:
	void ResetScene(Scene* scene) override;

	void AllocateSceneFromDepth(Scene* scene, const ITMView* view,
	                            const ITMTrackingState* trackingState,
	                            const ITMRenderState* renderState, bool onlyUpdateVisibleList = false,
	                            bool resetVisibleList = false) override;

	void FindVisibleBlocks(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                       ITMRenderState* renderState) override;

	explicit ITMSceneReconstructionEngine_CUDA(const std::shared_ptr<const ITMLibSettings>& settings);

	~ITMSceneReconstructionEngine_CUDA();

protected:
	void IntegrateIntoSceneVoxelProjection(Scene* scene,
	                                       const ITMView* view, const ITMTrackingState* trackingState,
	                                       const ITMRenderState* renderState) override;

	void IntegrateIntoSceneRayCasting(Scene* scene, const ITMView* view,
	                                  const ITMTrackingState* trackingState, const ITMRenderState* renderState) override;
};
}
