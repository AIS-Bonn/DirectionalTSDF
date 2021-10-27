// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <Utils/TemplateUtils.h>
#include "CPU/ITMSceneReconstructionEngine_CPU.h"

#ifndef COMPILE_WITHOUT_CUDA

#include "CUDA/ITMSceneReconstructionEngine_CUDA.h"

#endif

namespace ITMLib
{

/**
 * \brief This struct provides functions that can be used to construct scene reconstruction engines.
 */
struct ITMSceneReconstructionEngineFactory
{
	//#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

	/**
	 * \brief Makes a scene reconstruction engine.
	 *
	 * \param deviceType  The device on which the scene reconstruction engine should operate.
	 */
	static IITMSceneReconstructionEngine* MakeSceneReconstructionEngine(
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
};

}
