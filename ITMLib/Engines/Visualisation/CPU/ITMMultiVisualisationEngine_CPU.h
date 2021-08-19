// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMMultiVisualisationEngine.h"

namespace ITMLib
{
class ITMMultiVisualisationEngine_CPU : public ITMMultiVisualisationEngine
{
public:
	explicit ITMMultiVisualisationEngine_CPU(const std::shared_ptr<const ITMLibSettings>& settings)
		: ITMMultiVisualisationEngine(settings)
	{}

	~ITMMultiVisualisationEngine_CPU() = default;

	ITMRenderState* CreateRenderState(const Scene* scene, const Vector2i& imgSize) const;

	void PrepareRenderState(const ITMVoxelMapGraphManager<ITMVoxel>& sceneManager, ITMRenderState* state);

	void CreateExpectedDepths(const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                          ITMRenderState* renderState) const;

	void RenderImage(const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics, ITMRenderState* renderState,
	                 ITMUChar4Image* outputImage, IITMVisualisationEngine::RenderImageType type) const;
};
}

