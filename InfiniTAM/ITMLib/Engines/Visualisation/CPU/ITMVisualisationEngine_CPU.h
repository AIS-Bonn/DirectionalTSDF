// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMVisualisationEngine.h"

namespace ITMLib
{
template<class TVoxel, class TIndex>
class ITMVisualisationEngine_CPU_common : public ITMVisualisationEngine<TVoxel, TIndex>
{
public:
	explicit ITMVisualisationEngine_CPU_common(const std::shared_ptr<const ITMLibSettings>& settings)
		: ITMVisualisationEngine<TVoxel, TIndex>(settings)
	{}

	virtual ~ITMVisualisationEngine_CPU_common() = default;

	void FindSurface(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const ORUtils::SE3Pose* pose,
	                 const ITMIntrinsics* intrinsics, const ITMRenderState* renderState) const override;

	void RenderImage(const ITMScene <TVoxel, TIndex>* scene,
	                 const ORUtils::SE3Pose* pose,
	                 const ITMIntrinsics* intrinsics,
	                 const ITMRenderState* renderState,
	                 ITMUChar4Image* outputImage,
	                 IITMVisualisationEngine::RenderImageType type,
	                 IITMVisualisationEngine::RenderRaycastSelection raycastType) const override;

	void CreatePointCloud(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const ITMView* view,
	                      ITMTrackingState* trackingState, ITMRenderState* renderState, bool skipPoints) const override;

	void CreateICPMaps(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const ITMView* view,
	                   ITMTrackingState* trackingState, ITMRenderState* renderState) const override;

	void ForwardRender(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const ITMView* view,
	                   ITMTrackingState* trackingState, ITMRenderState* renderState) const override;

protected:
	void GenericRaycast(const ITMScene <TVoxel, TIndex>* scene, const Vector2i& imgSize, const Matrix4f& invM,
	                    const Vector4f& projParams, const ITMRenderState* renderState, bool updateVisibleList) const;
};

template<class TVoxel, class TIndex>
class ITMVisualisationEngine_CPU : public ITMVisualisationEngine_CPU_common<TVoxel, TIndex>
{
public:
	explicit ITMVisualisationEngine_CPU(const std::shared_ptr<const ITMLibSettings>& settings)
		: ITMVisualisationEngine_CPU_common<TVoxel, TIndex>(settings)
	{}

	~ITMVisualisationEngine_CPU() = default;

	ITMRenderState* CreateRenderState(const ITMScene <TVoxel, TIndex>* scene, const Vector2i& imgSize) const override;

	void FindVisibleBlocks(const ITMScene <TVoxel, TIndex>* scene, const ORUtils::SE3Pose* pose,
	                       const ITMIntrinsics* intrinsics, ITMRenderState* renderState) const override;

	int CountVisibleBlocks(const ITMScene <TVoxel, TIndex>* scene, const ITMRenderState* renderState, int minBlockId,
	                       int maxBlockId) const override;

	void CreateExpectedDepths(const ITMScene <TVoxel, TIndex>* scene, const ORUtils::SE3Pose* pose,
	                          const ITMIntrinsics* intrinsics, ITMRenderState* renderState) const override;
};

template<class TVoxel>
class ITMVisualisationEngine_CPU<TVoxel, ITMVoxelBlockHash>
	: public ITMVisualisationEngine_CPU_common<TVoxel, ITMVoxelBlockHash>
{
public:
	explicit ITMVisualisationEngine_CPU(const std::shared_ptr<const ITMLibSettings>& settings)
		: ITMVisualisationEngine_CPU_common<TVoxel, ITMVoxelBlockHash>(settings)
	{}

	~ITMVisualisationEngine_CPU() = default;

	ITMRenderState_VH*
	CreateRenderState(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const Vector2i& imgSize) const override;

	void FindVisibleBlocks(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const ORUtils::SE3Pose* pose,
	                       const ITMIntrinsics* intrinsics, ITMRenderState* renderState) const override;

	int CountVisibleBlocks(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const ITMRenderState* renderState,
	                       int minBlockId, int maxBlockId) const override;

	void CreateExpectedDepths(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const ORUtils::SE3Pose* pose,
	                          const ITMIntrinsics* intrinsics, ITMRenderState* renderState) const override;
};
}
