// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMSceneReconstructionEngine.h"
#include "../../../Objects/Scene/ITMPlainVoxelArray.h"

namespace ITMLib
{
	template<class TVoxel, class TIndex>
	class ITMSceneReconstructionEngine_CPU : public ITMSceneReconstructionEngine < TVoxel, TIndex >
	{
	public:
		explicit ITMSceneReconstructionEngine_CPU(ITMLibSettings::TSDFMode tsdfMode,
			ITMLibSettings::FusionMode fusionMode, ITMLibSettings::FusionMetric fusionMetric);
	};

template<class TVoxel>
	class ITMSceneReconstructionEngine_CPU<TVoxel, ITMVoxelBlockHash> : public ITMSceneReconstructionEngine < TVoxel, ITMVoxelBlockHash >
	{
	protected:
		ORUtils::MemoryBlock<HashEntryAllocType> *entriesAllocType;
		ORUtils::MemoryBlock<Vector4s> *blockCoords;

	public:
		void ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene);

		void AllocateSceneFromDepth(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view, const ITMTrackingState *trackingState,
			const ITMRenderState *renderState, bool onlyUpdateVisibleList = false, bool resetVisibleList = false);

		void IntegrateIntoScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view, const ITMTrackingState *trackingState,
			const ITMRenderState *renderState);

		ITMSceneReconstructionEngine_CPU(ITMLibSettings::TSDFMode tsdfMode,
			ITMLibSettings::FusionMode fusionMode, ITMLibSettings::FusionMetric fusionMetric);
		~ITMSceneReconstructionEngine_CPU();
	};

	template<class TVoxel>
	class ITMSceneReconstructionEngine_CPU<TVoxel, ITMPlainVoxelArray> : public ITMSceneReconstructionEngine < TVoxel, ITMPlainVoxelArray >
	{
	public:
		void ResetScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene);

		void AllocateSceneFromDepth(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view, const ITMTrackingState *trackingState,
			const ITMRenderState *renderState, bool onlyUpdateVisibleList = false, bool resetVisibleList = false);

		void IntegrateIntoScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view, const ITMTrackingState *trackingState,
			const ITMRenderState *renderState);
	};
}
