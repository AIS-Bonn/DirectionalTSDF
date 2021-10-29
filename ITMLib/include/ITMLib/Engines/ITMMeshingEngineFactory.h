// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <ITMLib/Engines/ITMMeshingEngine.h>
#include <ITMLib/Utils/ITMLibSettings.h>

namespace ITMLib
{

/**
 * \brief This struct provides functions that can be used to construct meshing engines.
 */
struct ITMMeshingEngineFactory
{
	//#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

	/**
	 * \brief Makes a meshing engine.
	 *
	 * \param deviceType  The device on which the meshing engine should operate.
	 */
	static ITMMeshingEngine* MakeMeshingEngine(ITMLibSettings::DeviceType deviceType);
};
}