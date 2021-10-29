// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include <ITMLib/Utils/ITMLibSettings.h>
#include <ITMLib/Engines/ITMLowLevelEngineFactory.h>

#include "CPU/ITMLowLevelEngine_CPU.h"

#ifndef COMPILE_WITHOUT_CUDA

#include "CUDA/ITMLowLevelEngine_CUDA.h"

#endif

namespace ITMLib
{

//#################### PUBLIC STATIC MEMBER FUNCTIONS ####################

ITMLowLevelEngine* ITMLowLevelEngineFactory::MakeLowLevelEngine(ITMLibSettings::DeviceType deviceType)
{
	ITMLowLevelEngine* lowLevelEngine = nullptr;

	switch (deviceType)
	{
		case ITMLibSettings::DEVICE_CPU:
			lowLevelEngine = new ITMLowLevelEngine_CPU();
			break;
		case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
			lowLevelEngine = new ITMLowLevelEngine_CUDA();
#endif
			break;
	}

	return lowLevelEngine;
}

}
