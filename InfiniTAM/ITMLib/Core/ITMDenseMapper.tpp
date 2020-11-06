// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMDenseMapper.h"

#include "ITMLib/Engines/Reconstruction/ITMSceneReconstructionEngineFactory.h"
#include "ITMLib/Engines/Swapping/ITMSwappingEngineFactory.h"
#include "ITMLib/Objects/RenderStates/ITMRenderState_VH.h"
#include "ITMLib/Utils/ITMTimer.h"
using namespace ITMLib;

ITMDenseMapper::ITMDenseMapper(const std::shared_ptr<const ITMLibSettings>& settings)
{
	sceneRecoEngine = ITMSceneReconstructionEngineFactory::MakeSceneReconstructionEngine(settings);
	swappingEngine = settings->swappingMode != ITMLibSettings::SWAPPINGMODE_DISABLED ? ITMSwappingEngineFactory::MakeSwappingEngine<ITMVoxel>(settings->deviceType) : nullptr;

	swappingMode = settings->swappingMode;
}

ITMDenseMapper::~ITMDenseMapper()
{
	delete sceneRecoEngine;
	delete swappingEngine;
}

void ITMDenseMapper::ResetScene(Scene *scene) const
{
	sceneRecoEngine->ResetScene(scene);
}

void ITMDenseMapper::ProcessFrame(const ITMView *view, const ITMTrackingState *trackingState, Scene *scene, ITMRenderState *renderState, bool resetVisibleList)
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

void ITMDenseMapper::UpdateVisibleList(const ITMView *view, const ITMTrackingState *trackingState, Scene *scene, ITMRenderState *renderState, bool resetVisibleList)
{
	sceneRecoEngine->AllocateSceneFromDepth(scene, view, trackingState, renderState, true, resetVisibleList);
}

const ITMSceneReconstructionEngine* ITMDenseMapper::GetSceneReconstructionEngine() const
{
	return sceneRecoEngine;
}
