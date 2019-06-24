// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMSceneReconstructionEngine.h"
#include "../../../Objects/Scene/ITMPlainVoxelArray.h"

namespace ITMLib
{
	template<class TVoxel, class TIndex>
	class ITMSceneReconstructionEngine_CUDA_common : public ITMSceneReconstructionEngine < TVoxel, TIndex >
	{
	public:
		explicit ITMSceneReconstructionEngine_CUDA_common(std::shared_ptr<const ITMLibSettings> settings);

	protected:
		void IntegrateIntoSceneRayCasting(ITMScene<TVoxel,TIndex> *scene, const ITMView *view,
																			const ITMTrackingState *trackingState, const ITMRenderState *renderState) override;
	};

template<class TVoxel, class TIndex>
	class ITMSceneReconstructionEngine_CUDA : public ITMSceneReconstructionEngine_CUDA_common < TVoxel, TIndex >
	{
	public:
		explicit ITMSceneReconstructionEngine_CUDA(std::shared_ptr<const ITMLibSettings> settings);
	};

template<class TVoxel>
	class ITMSceneReconstructionEngine_CUDA<TVoxel, ITMVoxelBlockHash> : public ITMSceneReconstructionEngine_CUDA_common < TVoxel, ITMVoxelBlockHash >
	{
	private:
		void *allocationTempData_device;
		void *allocationTempData_host;
		HashEntryAllocType *entriesAllocType_device;
		Vector4s *blockCoords_device;
		TSDFDirection *blockDirections_device;

	public:
		void ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene) override;

		void AllocateSceneFromDepth(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view, const ITMTrackingState *trackingState,
			const ITMRenderState *renderState, bool onlyUpdateVisibleList = false, bool resetVisibleList = false) override;

		explicit ITMSceneReconstructionEngine_CUDA(std::shared_ptr<const ITMLibSettings> settings);
		~ITMSceneReconstructionEngine_CUDA();

	protected:
		void IntegrateIntoSceneVoxelProjection(ITMScene<TVoxel, ITMVoxelBlockHash> *scene,
			const ITMView *view, const ITMTrackingState *trackingState, const ITMRenderState *renderState) override;
	};

	template<class TVoxel>
	class ITMSceneReconstructionEngine_CUDA<TVoxel, ITMPlainVoxelArray> : public ITMSceneReconstructionEngine_CUDA_common < TVoxel, ITMPlainVoxelArray >
	{
	public:
		void ResetScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene);

		void AllocateSceneFromDepth(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view, const ITMTrackingState *trackingState,
			const ITMRenderState *renderState, bool onlyUpdateVisibleList = false, bool resetVisibleList = false);

	protected:
		void IntegrateIntoSceneVoxelProjection(ITMScene<TVoxel, ITMPlainVoxelArray> *scene,
			const ITMView *view, const ITMTrackingState *trackingState, const ITMRenderState *renderState) override;
	};
}
