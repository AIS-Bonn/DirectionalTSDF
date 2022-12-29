//
// Created by Malte Splietker on 28.12.22.
//

#include <ITMLib/Utils/ITMLibSettings.h>
#include "ITMEvaluationEngineFactory.h"

#include "CPU/ITMEvaluationEngine_CPU.h"

#ifndef COMPILE_WITHOUT_CUDA

#include "CUDA/ITMEvaluationEngine_CUDA.h"

#endif

namespace ITMLib
{

//#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

ITMEvaluationEngine* ITMEvaluationEngineFactory::MakeEngine(ITMLibSettings::DeviceType deviceType)
{
	ITMEvaluationEngine* engine = nullptr;

	switch (deviceType)
	{
		case ITMLibSettings::DEVICE_CPU:
			engine = new ITMEvaluationEngine_CPU();
			break;
		case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
			engine = new ITMEvaluationEngine_CUDA();
#endif
			break;
	}

	return engine;
}

}
