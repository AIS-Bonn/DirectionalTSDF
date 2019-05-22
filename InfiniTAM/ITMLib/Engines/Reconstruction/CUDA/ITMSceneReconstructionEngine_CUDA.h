// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMSceneReconstructionEngine.h"
#include "../../../Objects/Scene/ITMPlainVoxelArray.h"

namespace ITMLib
{
	template<class TVoxel, class TIndex>
	class ITMSceneReconstructionEngine_CUDA : public ITMSceneReconstructionEngine < TVoxel, TIndex >
	{
	public:
		explicit ITMSceneReconstructionEngine_CUDA(ITMLibSettings::TSDFMode tsdfMode,
			ITMLibSettings::FusionMode fusionMode, ITMLibSettings::FusionMetric fusionMetric);
	};

template<class TVoxel>
	class ITMSceneReconstructionEngine_CUDA<TVoxel, ITMVoxelBlockHash> : public ITMSceneReconstructionEngine < TVoxel, ITMVoxelBlockHash >
	{
	private:
		void *allocationTempData_device;
		void *allocationTempData_host;
		HashEntryAllocType *entriesAllocType_device;
		Vector4s *blockCoords_device;
		TSDFDirection *blockDirections_device;

	public:
		void ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene);

		void AllocateSceneFromDepth(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view, const ITMTrackingState *trackingState,
			const ITMRenderState *renderState, bool onlyUpdateVisibleList = false, bool resetVisibleList = false);

		void IntegrateIntoScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view, const ITMTrackingState *trackingState,
			const ITMRenderState *renderState);

		explicit ITMSceneReconstructionEngine_CUDA(ITMLibSettings::TSDFMode tsdfMode,
			ITMLibSettings::FusionMode fusionMode, ITMLibSettings::FusionMetric fusionMetric);
		~ITMSceneReconstructionEngine_CUDA();
	};

	template<class TVoxel>
	class ITMSceneReconstructionEngine_CUDA<TVoxel, ITMPlainVoxelArray> : public ITMSceneReconstructionEngine < TVoxel, ITMPlainVoxelArray >
	{
	public:
		void ResetScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene);

		void AllocateSceneFromDepth(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view, const ITMTrackingState *trackingState,
			const ITMRenderState *renderState, bool onlyUpdateVisibleList = false, bool resetVisibleList = false);

		void IntegrateIntoScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view, const ITMTrackingState *trackingState,
			const ITMRenderState *renderState);
	};
}
