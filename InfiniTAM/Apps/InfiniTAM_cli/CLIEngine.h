// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMLib/Utils/ITMLibSettings.h"
#include "ITMLib/Utils/ITMImageTypes.h"
#include "ORUtils/FileUtils.h"
#include "ORUtils/NVTimer.h"

namespace ITMLib
{
	class ITMLoggingEngine;
	class ITMMainEngine;
	class ITMIMUMeasurement;
}

namespace InputSource
{
	class ImageSourceEngine;
	class IMUSourceEngine;
}

struct AppData;

namespace InfiniTAM::Engine
{
class CLIEngine
{
	static CLIEngine* instance;

	AppData *appData;
	ITMLib::ITMMainEngine* mainEngine;
	ITMLib::ITMLoggingEngine* statisticsEngine;

	StopWatchInterface* timer_instant;
	StopWatchInterface* timer_average;

private:
	ITMUChar4Image* inputRGBImage;
	ITMShortImage* inputRawDepthImage;
	ITMLib::ITMIMUMeasurement* inputIMUMeasurement;

	int currentFrameNo;
public:
	static CLIEngine* Instance()
	{
		if (not instance) instance = new CLIEngine();
		return instance;
	}

	void Initialise(AppData* appData, ITMLib::ITMMainEngine* mainEngine);

	void Shutdown();

	void Run();

	bool ProcessFrame();
};
}
