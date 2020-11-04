// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMSceneReconstructionEngine.h"

namespace ITMLib
{
class ITMSceneReconstructionEngine_CUDA : public ITMSceneReconstructionEngine
{
private:
	void *allocationTempData_device;
	void *allocationTempData_host;
	HashEntryAllocType *entriesAllocType_device;
	Vector4s *blockCoords_device;
	TSDFDirection *blockDirections_device;

public:
	void ResetScene(ITMScene<ITMVoxel, ITMVoxelIndex>* scene) override;

	void AllocateSceneFromDepth(ITMScene<ITMVoxel, ITMVoxelIndex>* scene, const ITMView* view,
	                            const ITMTrackingState* trackingState,
	                            const ITMRenderState* renderState, bool onlyUpdateVisibleList = false,
	                            bool resetVisibleList = false) override;

	explicit ITMSceneReconstructionEngine_CUDA(const std::shared_ptr<const ITMLibSettings>& settings);

	~ITMSceneReconstructionEngine_CUDA();

protected:
	void IntegrateIntoSceneVoxelProjection(ITMScene<ITMVoxel, ITMVoxelIndex>* scene,
	                                       const ITMView* view, const ITMTrackingState* trackingState,
	                                       const ITMRenderState* renderState) override;

	void IntegrateIntoSceneRayCasting(ITMScene<ITMVoxel, ITMVoxelIndex>* scene, const ITMView* view,
	                                  const ITMTrackingState* trackingState, const ITMRenderState* renderState) override;
};
}
