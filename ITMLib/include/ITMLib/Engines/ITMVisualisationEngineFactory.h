// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <memory>
#include <ITMLib/Engines/ITMVisualisationEngine.h>
#include <ITMLib/Utils/ITMLibSettings.h>

namespace ITMLib
{

/**
 * \brief This struct provides functions that can be used to construct visualisation engines.
 */
struct ITMVisualisationEngineFactory
{
	//#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

	/**
	 * \brief Makes a visualisation engine.
	 *
	 * \param deviceType  The device on which the visualisation engine should operate.
	 */
	static ITMVisualisationEngine* MakeVisualisationEngine(
		ITMLibSettings::DeviceType deviceType, const std::shared_ptr<const ITMLibSettings>& settings);
};

}
