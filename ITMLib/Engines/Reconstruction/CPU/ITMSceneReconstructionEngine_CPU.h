// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMSceneReconstructionEngine.h"

namespace ITMLib
{
template<typename TIndex>
class ITMSceneReconstructionEngine_CPU : public ITMSceneReconstructionEngine<TIndex>
{
protected:
	TSDF_CPU<TIndex, SummingVoxel>* summingVoxelMap = nullptr;

	/** Wrapper to make automatically getting tsdf/directionTSDF from scene object easier. */
	TSDF_CPU<TIndex, ITMVoxel>* GetTSDF(const Scene* scene);

	void IntegrateIntoSceneVoxelProjection(Scene* scene,
	                                       const ITMView* view, const ITMTrackingState* trackingState) override;

	void IntegrateIntoSceneRayCasting(Scene* scene, const ITMView* view,
	                                  const ITMTrackingState* trackingState) override;

public:
	void ResetScene(Scene* scene) override;

	void FindVisibleBlocks(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                       ITMRenderState* renderState) override;

	void AllocateSceneFromDepth(Scene* scene, const ITMView* view, const ITMTrackingState* trackingState) override;

	explicit ITMSceneReconstructionEngine_CPU(const std::shared_ptr<const ITMLibSettings>& settings);

	~ITMSceneReconstructionEngine_CPU();
};
}
