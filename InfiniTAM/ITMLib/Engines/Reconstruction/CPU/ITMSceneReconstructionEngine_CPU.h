// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMSceneReconstructionEngine.h"
#include "../../../Objects/Scene/ITMPlainVoxelArray.h"

namespace ITMLib
{
	template<class TVoxel, class TIndex>
	class ITMSceneReconstructionEngine_CPU_common : public ITMSceneReconstructionEngine < TVoxel, TIndex >
	{
	public:
		explicit ITMSceneReconstructionEngine_CPU_common(std::shared_ptr<const ITMLibSettings> settings);

	protected:
		void IntegrateIntoSceneRayCasting(ITMScene<TVoxel,TIndex> *scene, const ITMView *view,
		                                  const ITMTrackingState *trackingState, const ITMRenderState *renderState) override;
	};

template<class TVoxel, class TIndex>
	class ITMSceneReconstructionEngine_CPU : public ITMSceneReconstructionEngine_CPU_common < TVoxel, TIndex >
	{
	public:
		explicit ITMSceneReconstructionEngine_CPU(std::shared_ptr<const ITMLibSettings> settings);
	};

template<class TVoxel>
	class ITMSceneReconstructionEngine_CPU<TVoxel, ITMVoxelBlockHash> : public ITMSceneReconstructionEngine_CPU_common < TVoxel, ITMVoxelBlockHash >
	{
	protected:
		ORUtils::MemoryBlock<HashEntryAllocType> *entriesAllocType;
		ORUtils::MemoryBlock<Vector4s> *blockCoords;
		ORUtils::MemoryBlock<TSDFDirection> *blockDirections;

		void IntegrateIntoSceneVoxelProjection(ITMScene<TVoxel, ITMVoxelBlockHash> *scene,
			const ITMView *view, const ITMTrackingState *trackingState,
			const ITMRenderState *renderState) override;

	public:
		void ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene) override;

		void AllocateSceneFromDepth(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view, const ITMTrackingState *trackingState,
			const ITMRenderState *renderState, bool onlyUpdateVisibleList = false, bool resetVisibleList = false) override;

		explicit ITMSceneReconstructionEngine_CPU(std::shared_ptr<const ITMLibSettings> settings);
		~ITMSceneReconstructionEngine_CPU();
	};

	template<class TVoxel>
	class ITMSceneReconstructionEngine_CPU<TVoxel, ITMPlainVoxelArray> : public ITMSceneReconstructionEngine < TVoxel, ITMPlainVoxelArray >
	{
	public:
		void ResetScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene);

		void AllocateSceneFromDepth(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view, const ITMTrackingState *trackingState,
			const ITMRenderState *renderState, bool onlyUpdateVisibleList = false, bool resetVisibleList = false);

	protected:
		void IntegrateIntoSceneVoxelProjection(ITMScene<TVoxel, ITMPlainVoxelArray> *scene,
		                                       const ITMView *view, const ITMTrackingState *trackingState,
		                                       const ITMRenderState *renderState) override;
	};
}
