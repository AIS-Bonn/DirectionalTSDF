//
// Created by Malte Splietker on 28.10.21.
//

#include <ITMLib/Engines/ITMMeshingEngineFactory.h>

#include "CPU/ITMMeshingEngine_CPU.h"

#ifndef COMPILE_WITHOUT_CUDA

#include "CUDA/ITMMeshingEngine_CUDA.h"

#endif

using namespace ITMLib;

ITMMeshingEngine* ITMMeshingEngineFactory::MakeMeshingEngine(ITMLibSettings::DeviceType deviceType)
{
	ITMMeshingEngine* meshingEngine = nullptr;

	switch (deviceType)
	{
		case ITMLibSettings::DEVICE_CPU:
			meshingEngine = new ITMMeshingEngine_CPU;
			break;
		case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
			meshingEngine = new ITMMeshingEngine_CUDA;
#endif
			break;
	}

	return meshingEngine;
}
