// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMDenseMapper.h"

#include "ITMLib/Engines/Reconstruction/ITMSceneReconstructionEngineFactory.h"
#include "ITMLib/Engines/Swapping/ITMSwappingEngineFactory.h"
#include "ITMLib/Objects/RenderStates/ITMRenderState_VH.h"
#include "ITMLib/Utils/ITMTimer.h"
using namespace ITMLib;

template<class TVoxel, class TIndex>
ITMDenseMapper<TVoxel, TIndex>::ITMDenseMapper(const std::shared_ptr<const ITMLibSettings>& settings)
{
	sceneRecoEngine = ITMSceneReconstructionEngineFactory::MakeSceneReconstructionEngine<TVoxel,TIndex>(settings);
	swappingEngine = settings->swappingMode != ITMLibSettings::SWAPPINGMODE_DISABLED ? ITMSwappingEngineFactory::MakeSwappingEngine<TVoxel,TIndex>(settings->deviceType) : NULL;

	swappingMode = settings->swappingMode;
}

template<class TVoxel, class TIndex>
ITMDenseMapper<TVoxel,TIndex>::~ITMDenseMapper()
{
	delete sceneRecoEngine;
	delete swappingEngine;
}

template<class TVoxel, class TIndex>
void ITMDenseMapper<TVoxel,TIndex>::ResetScene(ITMScene<TVoxel,TIndex> *scene) const
{
	sceneRecoEngine->ResetScene(scene);
}

template<class TVoxel, class TIndex>
void ITMDenseMapper<TVoxel,TIndex>::ProcessFrame(const ITMView *view, const ITMTrackingState *trackingState, ITMScene<TVoxel,TIndex> *scene, ITMRenderState *renderState, bool resetVisibleList)
{
	sceneRecoEngine->GetTimeStats().Reset();

	// allocation
	sceneRecoEngine->AllocateSceneFromDepth(scene, view, trackingState, renderState, false, resetVisibleList);

	// integration
	sceneRecoEngine->IntegrateIntoScene(scene, view, trackingState, renderState);

	ITMTimer timer;
	timer.Tick();
	if (swappingEngine != NULL) {
		// swapping: CPU -> GPU
		if (swappingMode == ITMLibSettings::SWAPPINGMODE_ENABLED) swappingEngine->IntegrateGlobalIntoLocal(scene, renderState);

		// swapping: GPU -> CPU
		switch (swappingMode)
		{
		case ITMLibSettings::SWAPPINGMODE_ENABLED:
			swappingEngine->SaveToGlobalMemory(scene, renderState);
			break;
		case ITMLibSettings::SWAPPINGMODE_DELETE:
			swappingEngine->CleanLocalMemory(scene, renderState);
			break;
		case ITMLibSettings::SWAPPINGMODE_DISABLED:
			break;
		} 
	}
	sceneRecoEngine->GetTimeStats().swapping += timer.Tock();
}

template<class TVoxel, class TIndex>
void ITMDenseMapper<TVoxel,TIndex>::UpdateVisibleList(const ITMView *view, const ITMTrackingState *trackingState, ITMScene<TVoxel,TIndex> *scene, ITMRenderState *renderState, bool resetVisibleList)
{
	sceneRecoEngine->AllocateSceneFromDepth(scene, view, trackingState, renderState, true, resetVisibleList);
}

template<class TVoxel, class TIndex>
const ITMSceneReconstructionEngine* ITMDenseMapper<TVoxel, TIndex>::GetSceneReconstructionEngine() const
{
	return sceneRecoEngine;
}
