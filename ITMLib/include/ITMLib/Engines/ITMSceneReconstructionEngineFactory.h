// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <memory>
#include <ITMLib/Engines/ITMSceneReconstructionEngine.h>
#include <ITMLib/Utils/ITMLibSettings.h>

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
		const std::shared_ptr<const ITMLibSettings>& settings);
};

}
