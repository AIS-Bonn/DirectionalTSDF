//
// Created by Malte Splietker on 28.12.22.
//

#pragma once

#include <ITMLib/Engines/ITMEvaluationEngine.h>
#include <ITMLib/Utils/ITMLibSettings.h>

namespace ITMLib
{

/**
 * \brief This struct provides functions that can be used to construct evaluation engines.
 */
struct ITMEvaluationEngineFactory
{
	//#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

	/**
	 * \brief Makes a evaluation engine.
	 *
	 * \param deviceType  The device on which the low-level engine should operate.
	 */
	static ITMEvaluationEngine* MakeEngine(ITMLibSettings::DeviceType deviceType);
};

}
