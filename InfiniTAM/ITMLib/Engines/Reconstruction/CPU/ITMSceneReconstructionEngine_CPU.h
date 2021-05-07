// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMSceneReconstructionEngine.h"

namespace ITMLib
{
class SummingVoxelMap_CPU;

class ITMSceneReconstructionEngine_CPU : public ITMSceneReconstructionEngine
{
protected:
	ORUtils::MemoryBlock<HashEntryAllocType> *entriesAllocType;
	ORUtils::MemoryBlock<Vector4s> *blockCoords;
	ORUtils::MemoryBlock<TSDFDirection> *blockDirections;

	SummingVoxelMap_CPU* summingVoxelMap;

	void IntegrateIntoSceneVoxelProjection(Scene *scene,
		const ITMView *view, const ITMTrackingState *trackingState,
		const ITMRenderState *renderState) override;

	void IntegrateIntoSceneRayCasting(Scene *scene, const ITMView *view,
																		const ITMTrackingState *trackingState, const ITMRenderState *renderState) override;

public:
	void ResetScene(Scene *scene) override;

	void FindVisibleBlocks(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                       ITMRenderState* renderState) override;

	void AllocateSceneFromDepth(Scene *scene, const ITMView *view, const ITMTrackingState *trackingState,
		const ITMRenderState *renderState, bool onlyUpdateVisibleList = false, bool resetVisibleList = false) override;

	explicit ITMSceneReconstructionEngine_CPU(std::shared_ptr<const ITMLibSettings> settings);
	~ITMSceneReconstructionEngine_CPU();
};
}
