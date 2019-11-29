// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "CLIEngine.h"

#include <string.h>

#include "InputSource/ImageSourceEngine.h"
#include "InputSource/IMUSourceEngine.h"
#include "ITMLib/Engines/Logging/ITMLoggingEngine.h"
#include "ITMLib/Core/ITMMainEngine.h"
#include "../../ORUtils/FileUtils.h"

using namespace InfiniTAM::Engine;
using namespace InputSource;
using namespace ITMLib;

CLIEngine* CLIEngine::instance;

void CLIEngine::Initialise(ImageSourceEngine* imageSource_, IMUSourceEngine* imuSource_,
                           ITMMainEngine* mainEngine_, ITMLibSettings::DeviceType deviceType,
                           const std::string& outFolder)
{
	this->imageSource = imageSource_;
	this->imuSource = imuSource_;
	this->mainEngine = mainEngine_;

	this->currentFrameNo = 0;

	bool allocateGPU = false;
	if (deviceType == ITMLibSettings::DEVICE_CUDA) allocateGPU = true;

	inputRGBImage = new ITMUChar4Image(imageSource->getRGBImageSize(), true, allocateGPU);
	inputRawDepthImage = new ITMShortImage(imageSource->getDepthImageSize(), true, allocateGPU);
	inputIMUMeasurement = new ITMIMUMeasurement();
	statisticsEngine = new ITMLoggingEngine();

	this->statisticsEngine->Initialize(outFolder);

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
	if (!imageSource->hasMoreImages()) return false;
	imageSource->getImages(inputRGBImage, inputRawDepthImage);

	if (imuSource != NULL)
	{
		if (!imuSource->hasMoreMeasurements()) return false;
		else imuSource->getMeasurement(inputIMUMeasurement);
	}

	sdkResetTimer(&timer_instant);
	sdkStartTimer(&timer_instant);
	sdkStartTimer(&timer_average);

	//actual processing on the mailEngine
	if (imuSource != NULL) mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage, inputIMUMeasurement);
	else mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage);

#ifndef COMPILE_WITHOUT_CUDA
	ORcudaSafeCall(cudaThreadSynchronize());
#endif
	sdkStopTimer(&timer_instant);
	sdkStopTimer(&timer_average);

	statisticsEngine->LogTimeStats(mainEngine->GetTimeStats());
	statisticsEngine->LogPose(*mainEngine->GetTrackingState());

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
