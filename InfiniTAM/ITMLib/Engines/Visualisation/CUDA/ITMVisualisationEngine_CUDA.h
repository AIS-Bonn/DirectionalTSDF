// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <memory>
#include "../Interface/ITMVisualisationEngine.h"

struct RenderingBlock;

namespace ITMLib
{
/** Common functions for all specializations */
template<class TVoxel, class TIndex>
class ITMVisualisationEngine_CUDA_common : public ITMVisualisationEngine<TVoxel, TIndex>
{
public:
	explicit ITMVisualisationEngine_CUDA_common(std::shared_ptr<const ITMLibSettings> settings);

	virtual ~ITMVisualisationEngine_CUDA_common() = default;

	void CreatePointCloud(const ITMScene <TVoxel, TIndex>* scene, const ITMView* view,
	                      ITMTrackingState* trackingState, ITMRenderState* renderState,
	                      bool skipPoints) const override;

	void ForwardRender(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const ITMView* view,
	                   ITMTrackingState* trackingState, ITMRenderState* renderState) const override;

	void RenderImage(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const ORUtils::SE3Pose* pose,
	                 const ITMIntrinsics* intrinsics, const ITMRenderState* renderState,
	                 ITMUChar4Image* outputImage,
	                 IITMVisualisationEngine::RenderImageType type,
	                 IITMVisualisationEngine::RenderRaycastSelection raycastType) const override;

	void FindSurface(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const ORUtils::SE3Pose* pose,
	                 const ITMIntrinsics* intrinsics, const ITMRenderState* renderState) const override;

	void
	CreateICPMaps(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const ITMView* view, ITMTrackingState* trackingState,
	              ITMRenderState* renderState) const override;

protected:
	uint* noTotalPoints_device;

	void GenericRaycast(const ITMScene <TVoxel, TIndex>* scene, const Vector2i& imgSize, const Matrix4f& invM,
	                    const Vector4f& projParams, const ITMRenderState* renderState, bool updateVisibleList) const;
};

template<class TVoxel, class TIndex>
class ITMVisualisationEngine_CUDA : public ITMVisualisationEngine_CUDA_common<TVoxel, TIndex>
{
private:

public:
	explicit ITMVisualisationEngine_CUDA(std::shared_ptr<const ITMLibSettings> settings);

	~ITMVisualisationEngine_CUDA();

	ITMRenderState* CreateRenderState(const ITMScene <TVoxel, TIndex>* scene, const Vector2i& imgSize) const;

	void FindVisibleBlocks(const ITMScene <TVoxel, TIndex>* scene, const ORUtils::SE3Pose* pose,
	                       const ITMIntrinsics* intrinsics, ITMRenderState* renderState) const;

	int CountVisibleBlocks(const ITMScene <TVoxel, TIndex>* scene, const ITMRenderState* renderState, int minBlockId,
	                       int maxBlockId) const;

	void CreateExpectedDepths(const ITMScene <TVoxel, TIndex>* scene, const ORUtils::SE3Pose* pose,
	                          const ITMIntrinsics* intrinsics, ITMRenderState* renderState) const;
};

template<class TVoxel>
class ITMVisualisationEngine_CUDA<TVoxel, ITMVoxelBlockHash>
	: public ITMVisualisationEngine_CUDA_common<TVoxel, ITMVoxelBlockHash>
{
private:
	RenderingBlock* renderingBlockList_device;
	uint* noTotalBlocks_device;
	int* noVisibleEntries_device;
public:
	explicit ITMVisualisationEngine_CUDA(std::shared_ptr<const ITMLibSettings> settings);

	~ITMVisualisationEngine_CUDA();

	ITMRenderState_VH*
	CreateRenderState(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const Vector2i& imgSize) const;

	void FindVisibleBlocks(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const ORUtils::SE3Pose* pose,
	                       const ITMIntrinsics* intrinsics, ITMRenderState* renderState) const;

	int CountVisibleBlocks(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const ITMRenderState* renderState,
	                       int minBlockId, int maxBlockId) const;

	void CreateExpectedDepths(const ITMScene <TVoxel, ITMVoxelBlockHash>* scene, const ORUtils::SE3Pose* pose,
	                          const ITMIntrinsics* intrinsics, ITMRenderState* renderState) const;
};
}
