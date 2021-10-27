// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMSceneReconstructionEngine.h"

namespace ITMLib
{
template<typename TIndex>
class ITMSceneReconstructionEngine_CUDA : public ITMSceneReconstructionEngine<TIndex>
{
private:
	TSDF_CUDA<TIndex, SummingVoxel>* summingVoxelMap = nullptr;

	/** Wrapper to make automatically getting tsdf/directionTSDF from scene object easier. */
	TSDF_CUDA<TIndex, ITMVoxel>* GetTSDF(const Scene* scene);

public:
	void ResetScene(Scene* scene) override;

	void AllocateSceneFromDepth(Scene* scene, const ITMView* view,
	                            const ITMTrackingState* trackingState) override;

	void FindVisibleBlocks(const Scene* scene, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                       ITMRenderState* renderState) override;

	explicit ITMSceneReconstructionEngine_CUDA(const std::shared_ptr<const ITMLibSettings>& settings);

	~ITMSceneReconstructionEngine_CUDA() override;

protected:
	void IntegrateIntoSceneVoxelProjection(Scene* scene, const ITMView* view,
	                                       const ITMTrackingState* trackingState) override;

	void IntegrateIntoSceneRayCasting(Scene* scene, const ITMView* view,
	                                  const ITMTrackingState* trackingState) override;
};
}