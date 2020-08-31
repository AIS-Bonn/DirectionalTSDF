// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "CLIEngine.h"

#include <string.h>
#include <Apps/Utils/CLIUtils.h>

#include "InputSource/ImageSourceEngine.h"
#include "InputSource/IMUSourceEngine.h"
#include "ITMLib/Engines/Logging/ITMLoggingEngine.h"
#include "ITMLib/Core/ITMMainEngine.h"
#include "../../ORUtils/FileUtils.h"

using namespace InfiniTAM::Engine;
using namespace InputSource;
using namespace ITMLib;

CLIEngine* CLIEngine::instance;

void CLIEngine::Run()
{
	while (true)
	{
		if (!ProcessFrame()) break;
	}
}

void CLIEngine::Shutdown()
{
	if (appData->numberExportPointClouds == 0)
	{
		CollectPointClouds();
	} else if (appData->numberExportPointClouds > 0)
	{
		int N = currentFrameNo / appData->numberExportPointClouds;
		CollectPointClouds(N);
	}

	sdkDeleteTimer(&timer_instant);
	sdkDeleteTimer(&timer_average);

	statisticsEngine.CloseAll();

	delete instance;
}

bool CLIEngine::_processFrame()
{
	//actual processing on the mailEngine
	if (appData->imuSource != nullptr) mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage, inputIMUMeasurement, inputPose);
	else mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage, nullptr, inputPose);

#ifndef COMPILE_WITHOUT_CUDA
	ORcudaSafeCall(cudaThreadSynchronize());
#endif

	float processedTime_inst = sdkGetTimerValue(&timer_instant);
	float processedTime_avg = sdkGetAverageTimerValue(&timer_average);

	printf("frame %04i: time %.2f, avg %.2f, tracking: %s\n",
	       currentFrameNo, processedTime_inst, processedTime_avg,
	       ITMTrackingState::TrackingResultToString(mainEngine->GetTrackingState()->trackerResult).c_str());

	return true;
}

void CLIEngine::_initialise(int argc, char** argv, AppData* appData, ITMLib::ITMMainEngine* mainEngine)
{

}
