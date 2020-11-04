//
// Created by Malte Splietker on 25.08.20.
//

#pragma once

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

#include <Apps/Utils/CLIUtils.h>
#include <ITMLib/Core/ITMMainEngine.h>
#include <ITMLib/Engines/Logging/ITMLoggingEngine.h>
#include <ITMLib/Engines/ViewBuilding/Interface/ITMViewBuilder.h>
#include <ITMLib/Engines/ViewBuilding/ITMViewBuilderFactory.h>
#include <ORUtils/NVTimer.h>
#include <ORUtils/FileUtils.h>

namespace InfiniTAM
{
namespace Engine
{
class AppEngine
{

protected:
	AppData* appData;
	ITMLib::ITMMainEngine* mainEngine;
	ITMLib::ITMLoggingEngine statisticsEngine;

	ITMUChar4Image *inputRGBImage; ITMShortImage *inputRawDepthImage;
	const ORUtils::SE3Pose *inputPose;
	ITMLib::ITMIMUMeasurement *inputIMUMeasurement;

	StopWatchInterface *timer_instant;
	StopWatchInterface *timer_average;

	char *outFolder;

	int currentFrameNo;
	bool normalsFromSDF;

//	void CollectICPErrorImages();
	std::vector<std::pair<ITMUChar4Image*, ITMShortImage*>> inputImages;
	std::vector<ORUtils::SE3Pose> trackingPoses;

	inline void CollectICPErrorImages()
	{
		if (inputImages.empty())
			return;

		ITMLib::ITMView* view = nullptr;
		ITMViewBuilder* viewBuilder = ITMViewBuilderFactory::MakeViewBuilder(appData->imageSource->getCalib(),
		                                                                     appData->internalSettings->deviceType);
		// needs to be called once with view=nullptr for initialization
		viewBuilder->UpdateView(&view, inputImages.front().first, inputImages.front().second,
		                        appData->internalSettings->useBilateralFilter);
		view = mainEngine->GetView();

		ITMUChar4Image* outputImage = new ITMUChar4Image(inputImages.front().first->noDims, true, false);
		char str[250];

		int lastPercentile = -1;
		for (size_t i = 0; i < inputImages.size(); i++)
		{
			int percentile = (i * 10) / inputImages.size();
			if (percentile != lastPercentile)
			{
				printf("%i%%\t", percentile * 10);
				lastPercentile = percentile;
			}
			ITMUChar4Image* rgbImage = inputImages.at(i).first;
			ITMShortImage* depthImage = inputImages.at(i).second;

			viewBuilder->UpdateView(&view, rgbImage, depthImage,
			                        appData->internalSettings->useBilateralFilter);

			ORUtils::SE3Pose* pose = &trackingPoses.at(i);
			mainEngine->GetTrackingState()->pose_d->SetFrom(pose);

			mainEngine->GetImage(outputImage, ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_ICP_ERROR,
			                     pose, &view->calib.intrinsics_d, normalsFromSDF);

			sprintf(str, "%s/recording/error_%04zu.ppm", outFolder, i);
			SaveImageToFile(outputImage, str);
		}

		free(outputImage);
		free(viewBuilder);
	}

	/**
	 * Collect and store point cloud renderings from every N-th pose
	 * @param N
	 */
	inline void CollectPointClouds(int N=1)
	{
		if (inputImages.empty())
			return;

		ITMLib::ITMView* view = nullptr;
		ITMViewBuilder* viewBuilder = ITMViewBuilderFactory::MakeViewBuilder(appData->imageSource->getCalib(),
		                                                                     appData->internalSettings->deviceType);
		// needs to be called once with view=nullptr for initialization
		viewBuilder->UpdateView(&view, inputImages.front().first, inputImages.front().second,
		                        appData->internalSettings->useBilateralFilter);
		view = mainEngine->GetView();


		ITMUChar4Image* outputImage = new ITMUChar4Image(inputImages.front().first->noDims, true, false);
		char str[250];

		ORUtils::Image<ORUtils::Vector4<float>>* points = new ORUtils::Image<ORUtils::Vector4<float>>(
			mainEngine->GetView()->calib.intrinsics_d.imgSize, true, true);
		ORUtils::Image<ORUtils::Vector4<float>>* normals = new ORUtils::Image<ORUtils::Vector4<float>>(
			mainEngine->GetView()->calib.intrinsics_d.imgSize, true, true);

		int lastPercentile = -1;
		for (size_t i = 0; i < inputImages.size(); i += N)
		{
			int percentile = (i * 10) / inputImages.size();
			if (percentile != lastPercentile)
			{
				printf("%i%%\t", percentile * 10);
				lastPercentile = percentile;
			}
			ITMUChar4Image* rgbImage = inputImages.at(i).first;
			ITMShortImage* depthImage = inputImages.at(i).second;

			viewBuilder->UpdateView(&view, rgbImage, depthImage,
			                        appData->internalSettings->useBilateralFilter);

			ORUtils::SE3Pose* pose = &trackingPoses.at(i);
			mainEngine->GetTrackingState()->pose_d->SetFrom(pose);

			mainEngine->GetImage(outputImage, ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_ICP_ERROR,
			                     pose, &mainEngine->GetView()->calib.intrinsics_d, normalsFromSDF);


			if (appData->internalSettings->deviceType == ITMLibSettings::DEVICE_CUDA)
			{
				points->SetFrom(mainEngine->GetTrackingState()->pointCloud->locations,
				                ORUtils::MemoryBlock<Vector4f>::CUDA_TO_CPU);
				normals->SetFrom(mainEngine->GetTrackingState()->pointCloud->locations,
				                ORUtils::MemoryBlock<Vector4f>::CUDA_TO_CPU);
			} else{
				points->SetFrom(mainEngine->GetTrackingState()->pointCloud->locations,
				                ORUtils::MemoryBlock<Vector4f>::CPU_TO_CPU);
				normals->SetFrom(mainEngine->GetTrackingState()->pointCloud->locations,
				                 ORUtils::MemoryBlock<Vector4f>::CPU_TO_CPU);
			}

		sprintf(str, "%s/cloud_%04zu.pcd", outFolder, i);
			SavePointCloudToPCL(
				points->GetData(MEMORYDEVICE_CPU), normals->GetData(MEMORYDEVICE_CPU),
				inputImages.front().second->noDims, *pose, str);
		}

		free(points);
		free(normals);
		free(outputImage);
		free(viewBuilder);
	}

	virtual void _initialise(int argc, char** argv, AppData* appData, ITMMainEngine *mainEngine) = 0;
	virtual bool _processFrame() = 0;

public:
	AppEngine()
		: normalsFromSDF(false)
	{}

	virtual ~AppEngine()
	{
		sdkDeleteTimer(&timer_instant);
		sdkDeleteTimer(&timer_average);

		for (auto imgs: inputImages)
		{
//		delete imgs.first;
			delete imgs.second;
		}

		delete inputRGBImage;
		delete inputRawDepthImage;
		delete inputIMUMeasurement;
	}

	inline bool ProcessFrame()
	{
		if (!appData->imageSource->hasMoreImages()) return false;
		appData->imageSource->getImages(inputRGBImage, inputRawDepthImage);

		if (appData->imuSource != nullptr) {
			if (!appData->imuSource->hasMoreMeasurements()) return false;
			else appData->imuSource->getMeasurement(inputIMUMeasurement);
		}

		inputPose = nullptr;
		if (appData->trajectorySource != nullptr)
		{
			if (!appData->trajectorySource->hasMorePoses()) return false;
			inputPose = appData->trajectorySource->getPose();
		} else if (currentFrameNo == 0)
		{
			inputPose = &appData->initialPose;
		}

		// Safe input images
		inputImages.emplace_back();
//	inputImages.back().first = new ITMUChar4Image(true, false);
//	inputImages.back().first->SetFrom(inputRGBImage, ITMUChar4Image::CPU_TO_CPU);
		inputImages.back().first = inputRGBImage; // not required for error renderings, so only store reference
		inputImages.back().second = new ITMShortImage(true, false);
		inputImages.back().second->SetFrom(inputRawDepthImage, ITMShortImage::CPU_TO_CPU);
		trackingPoses.push_back(*mainEngine->GetTrackingState()->pose_d);

		sdkResetTimer(&timer_instant);
		sdkStartTimer(&timer_instant);
		sdkStartTimer(&timer_average);

		if (not _processFrame()) return false;

		sdkStopTimer(&timer_instant); sdkStopTimer(&timer_average);

		statisticsEngine.LogTimeStats(mainEngine->GetTimeStats());
		statisticsEngine.LogPose(*mainEngine->GetTrackingState());
//		statisticsEngine.LogBlockAllocations(mainEngine->GetAllocationsPerDirection());

		currentFrameNo++;

		return true;
	}

	inline void Initialise(int argc, char** argv, AppData* appData, ITMMainEngine *mainEngine)
	{
		this->appData = appData;
		this->mainEngine = mainEngine;

		this->currentFrameNo = 0;

		// Initialize output directtory
		size_t len = appData->outputDirectory.size();
		this->outFolder = new char[len + 1];
		strcpy(this->outFolder, appData->outputDirectory.c_str());
		fs::create_directories(appData->outputDirectory);

		this->statisticsEngine.Initialize(std::string(outFolder));
		bool allocateGPU = appData->internalSettings->deviceType == ITMLibSettings::DEVICE_CUDA;
		inputRGBImage = new ITMUChar4Image(appData->imageSource->getRGBImageSize(), true, allocateGPU);
		inputRawDepthImage = new ITMShortImage(appData->imageSource->getDepthImageSize(), true, allocateGPU);
		inputIMUMeasurement = new ITMIMUMeasurement();

		sdkCreateTimer(&timer_instant);
		sdkCreateTimer(&timer_average);
		sdkResetTimer(&timer_average);

		_initialise(argc, argv, appData, mainEngine);

		#ifndef COMPILE_WITHOUT_CUDA
				ORcudaSafeCall(cudaThreadSynchronize());
		#endif

		printf("initialised.\n");
	}
};



} // namespace Engine
} // namespace InfiniTAM
