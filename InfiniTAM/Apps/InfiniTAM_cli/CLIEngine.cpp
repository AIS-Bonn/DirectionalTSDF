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

void CLIEngine::Initialise(AppData* appData, ITMMainEngine* mainEngine)
{
	this->appData = appData;
	this->mainEngine = mainEngine;

	this->currentFrameNo = 0;

	bool allocateGPU = false;
	if (appData->internalSettings->deviceType == ITMLibSettings::DEVICE_CUDA) allocateGPU = true;

	inputRGBImage = new ITMUChar4Image(appData->imageSource->getRGBImageSize(), true, allocateGPU);
	inputRawDepthImage = new ITMShortImage(appData->imageSource->getDepthImageSize(), true, allocateGPU);
	inputIMUMeasurement = new ITMIMUMeasurement();
	statisticsEngine = new ITMLoggingEngine();

	this->statisticsEngine->Initialize(appData->outputDirectory);

#ifndef COMPILE_WITHOUT_CUDA
	ORcudaSafeCall(cudaThreadSynchronize());
#endif

	sdkCreateTimer(&timer_instant);
	sdkCreateTimer(&timer_average);

	sdkResetTimer(&timer_average);

	printf("initialised.\n");
}

bool CLIEngine::ProcessFrame()
{
	if (!appData->imageSource->hasMoreImages()) return false;
	appData->imageSource->getImages(inputRGBImage, inputRawDepthImage);

	if (appData->imuSource != nullptr)
	{
		if (!appData->imuSource->hasMoreMeasurements()) return false;
		else appData->imuSource->getMeasurement(inputIMUMeasurement);
	}

	const ORUtils::SE3Pose *inputPose = nullptr;
	if (appData->trajectorySource != nullptr)
	{
		if (!appData->trajectorySource->hasMorePoses()) return false;
		inputPose = appData->trajectorySource->getPose();
	}

	sdkResetTimer(&timer_instant);
	sdkStartTimer(&timer_instant);
	sdkStartTimer(&timer_average);

	//actual processing on the mailEngine
	if (appData->imuSource != nullptr) mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage, inputIMUMeasurement, inputPose);
	else mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage, nullptr, inputPose);

#ifndef COMPILE_WITHOUT_CUDA
	ORcudaSafeCall(cudaThreadSynchronize());
#endif
	sdkStopTimer(&timer_instant);
	sdkStopTimer(&timer_average);

	statisticsEngine->LogTimeStats(mainEngine->GetTimeStats());
	statisticsEngine->LogPose(*mainEngine->GetTrackingState());
	statisticsEngine->LogBlockAllocations(mainEngine->GetAllocationsPerDirection());

	float processedTime_inst = sdkGetTimerValue(&timer_instant);
	float processedTime_avg = sdkGetAverageTimerValue(&timer_average);

	printf("frame %04i: time %.2f, avg %.2f, tracking: %s\n",
	       currentFrameNo, processedTime_inst, processedTime_avg,
	       ITMTrackingState::TrackingResultToString(mainEngine->GetTrackingState()->trackerResult).c_str());

	currentFrameNo++;

	return true;
}

void CLIEngine::Run()
{
	while (true)
	{
		if (!ProcessFrame()) break;
	}
}

void CLIEngine::Shutdown()
{
	sdkDeleteTimer(&timer_instant);
	sdkDeleteTimer(&timer_average);

	statisticsEngine->CloseAll();

	delete inputRGBImage;
	delete inputRawDepthImage;
	delete inputIMUMeasurement;

	delete statisticsEngine;

	delete instance;
}
