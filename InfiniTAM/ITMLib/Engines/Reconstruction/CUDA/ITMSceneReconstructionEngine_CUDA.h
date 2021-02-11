// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMSceneReconstructionEngine.h"

namespace ITMLib
{
class SummingVoxelMap_CUDA;

class ITMSceneReconstructionEngine_CUDA : public ITMSceneReconstructionEngine
{
private:
	void *allocationTempData_device;
	void *allocationTempData_host;
	HashEntryAllocType *entriesAllocType_device;
	Vector4s *blockCoords_device;
	TSDFDirection *blockDirections_device;

	SummingVoxelMap_CUDA* summingVoxelMap;

public:
	void ResetScene(Scene* scene) override;

	void AllocateSceneFromDepth(Scene* scene, const ITMView* view,
	                            const ITMTrackingState* trackingState,
	                            const ITMRenderState* renderState, bool onlyUpdateVisibleList = false,
	                            bool resetVisibleList = false) override;

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
