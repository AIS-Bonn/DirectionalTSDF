//
// Created by Malte Splietker on 28.10.21.
//

#include <ITMLib/Engines/ITMSceneReconstructionEngineFactory.h>

#include "CPU/ITMSceneReconstructionEngine_CPU.h"

#ifndef COMPILE_WITHOUT_CUDA

#include "CUDA/ITMSceneReconstructionEngine_CUDA.h"

#endif

using namespace ITMLib;

IITMSceneReconstructionEngine* ITMSceneReconstructionEngineFactory::MakeSceneReconstructionEngine(
	const std::shared_ptr<const ITMLibSettings>& settings)
{
	IITMSceneReconstructionEngine* sceneRecoEngine = nullptr;

	switch (settings->deviceType)
	{
		case ITMLibSettings::DEVICE_CPU:
			if (settings->Directional())
				sceneRecoEngine = new ITMSceneReconstructionEngine_CPU<ITMIndexDirectional>(settings);
			else sceneRecoEngine = new ITMSceneReconstructionEngine_CPU<ITMIndex>(settings);
			break;
		case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
			if (settings->Directional())
				sceneRecoEngine = new ITMSceneReconstructionEngine_CUDA<ITMIndexDirectional>(settings);
			else sceneRecoEngine = new ITMSceneReconstructionEngine_CUDA<ITMIndex>(settings);
#endif
			break;
	}

	return sceneRecoEngine;
}
