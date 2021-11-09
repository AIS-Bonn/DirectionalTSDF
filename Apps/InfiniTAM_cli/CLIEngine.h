// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <Apps/AppEngine/AppEngine.h>
#include <ITMLib/Utils/ITMLibSettings.h>
#include <ITMLib/Utils/ITMImageTypes.h>
#include <ORUtils/FileUtils.h>
#include <ORUtils/NVTimer.h>

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
class CLIEngine : public AppEngine
{

private:
	void _initialise(int argc, char**argv, AppData* appData, ITMLib::ITMMainEngine* mainEngine) override;
	bool _postFusion() override;

	static CLIEngine* instance;

public:
	static CLIEngine* Instance()
	{
		if (not instance) instance = new CLIEngine();
		return instance;
	}

	void Shutdown();

	void Run();

//	bool ProcessFrame();
};
}
