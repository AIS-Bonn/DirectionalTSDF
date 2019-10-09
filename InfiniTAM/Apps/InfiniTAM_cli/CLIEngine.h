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

namespace InfiniTAM::Engine
{
class CLIEngine
{
	static CLIEngine* instance;

	InputSource::ImageSourceEngine* imageSource;
	InputSource::IMUSourceEngine* imuSource;
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

	void Initialise(InputSource::ImageSourceEngine* imageSource_, InputSource::IMUSourceEngine* imuSource_,
	                ITMLib::ITMMainEngine* mainEngine_,
	                ITMLib::ITMLibSettings::DeviceType deviceType, const std::string& outFolder);

	void Shutdown();

	void Run();

	bool ProcessFrame();
};
}
