//
// Created by Malte Splietker on 28.10.21.
//

#include <ITMLib/Engines/ITMVisualisationEngineFactory.h>

#include "CPU/ITMVisualisationEngine_CPU.h"

#ifndef COMPILE_WITHOUT_CUDA

#include "CUDA/ITMVisualisationEngine_CUDA.h"

#endif

using namespace ITMLib;

ITMVisualisationEngine* ITMVisualisationEngineFactory::MakeVisualisationEngine(
	ITMLibSettings::DeviceType deviceType, const std::shared_ptr<const ITMLibSettings>& settings)
{
	ITMVisualisationEngine* visualisationEngine = nullptr;

	switch (deviceType)
	{
		case ITMLibSettings::DEVICE_CPU:
			visualisationEngine = new ITMVisualisationEngine_CPU(settings);
			break;
		case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
			visualisationEngine = new ITMVisualisationEngine_CUDA(settings);
#endif
			break;
	}

	return visualisationEngine;
}
