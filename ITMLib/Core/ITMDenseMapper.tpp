// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMDenseMapper.h"

#include "ITMLib/Engines/Reconstruction/ITMSceneReconstructionEngineFactory.h"
#include "ITMLib/Objects/RenderStates/ITMRenderState.h"
#include "ITMLib/Utils/ITMTimer.h"

using namespace ITMLib;

ITMDenseMapper::ITMDenseMapper(const std::shared_ptr<const ITMLibSettings>& settings)
{
	sceneRecoEngine = ITMSceneReconstructionEngineFactory::MakeSceneReconstructionEngine(settings);

	swappingMode = settings->swappingMode;
}

ITMDenseMapper::~ITMDenseMapper()
{
	delete sceneRecoEngine;
}

void ITMDenseMapper::ResetScene(Scene* scene) const
{
	sceneRecoEngine->ResetScene(scene);
}

void ITMDenseMapper::ProcessFrame(const ITMView* view, const ITMTrackingState* trackingState, Scene* scene,
                                  ITMRenderState* renderState)
{
	sceneRecoEngine->GetTimeStats().Reset();

	// allocation
	sceneRecoEngine->AllocateSceneFromDepth(scene, view, trackingState);

	sceneRecoEngine->FindVisibleBlocks(scene, trackingState->pose_d, &view->calib.intrinsics_d, renderState);

	// integration
	sceneRecoEngine->IntegrateIntoScene(scene, view, trackingState);

	ITMTimer timer;
	timer.Tick();
	if (swappingMode != ITMLibSettings::SWAPPINGMODE_DISABLED)
	{
	}
	sceneRecoEngine->GetTimeStats().swapping += timer.Tock();
}

void ITMDenseMapper::UpdateVisibleList(const ITMView* view, const ITMTrackingState* trackingState, Scene* scene,
                                       ITMRenderState* renderState)
{
	sceneRecoEngine->FindVisibleBlocks(scene, trackingState->pose_d, &(view->calib.intrinsics_d), renderState);
}