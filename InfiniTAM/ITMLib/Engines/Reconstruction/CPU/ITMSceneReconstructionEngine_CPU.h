// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMSceneReconstructionEngine.h"

namespace ITMLib
{
class ITMSceneReconstructionEngine_CPU : public ITMSceneReconstructionEngine
{
protected:
	ORUtils::MemoryBlock<HashEntryAllocType> *entriesAllocType;
	ORUtils::MemoryBlock<Vector4s> *blockCoords;
	ORUtils::MemoryBlock<TSDFDirection> *blockDirections;

	void IntegrateIntoSceneVoxelProjection(ITMScene<ITMVoxel, ITMVoxelIndex> *scene,
		const ITMView *view, const ITMTrackingState *trackingState,
		const ITMRenderState *renderState) override;

	void IntegrateIntoSceneRayCasting(ITMScene<ITMVoxel,ITMVoxelIndex> *scene, const ITMView *view,
																		const ITMTrackingState *trackingState, const ITMRenderState *renderState) override;

public:
	void ResetScene(ITMScene<ITMVoxel, ITMVoxelIndex> *scene) override;

	void AllocateSceneFromDepth(ITMScene<ITMVoxel, ITMVoxelIndex> *scene, const ITMView *view, const ITMTrackingState *trackingState,
		const ITMRenderState *renderState, bool onlyUpdateVisibleList = false, bool resetVisibleList = false) override;

	explicit ITMSceneReconstructionEngine_CPU(std::shared_ptr<const ITMLibSettings> settings);
	~ITMSceneReconstructionEngine_CPU();
};
}
