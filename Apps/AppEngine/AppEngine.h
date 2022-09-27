//
// Created by Malte Splietker on 25.08.20.
//

#pragma once

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

#include <Apps/Utils/CLIUtils.h>
#include <ITMLib/Core/ITMMainEngine.h>
#include <ITMLib/Engines/ITMLoggingEngine.h>
#include <ITMLib/Engines/ITMViewBuilder.h>
#include <ITMLib/Engines/ITMViewBuilderFactory.h>
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

	ITMUChar4Image* inputRGBImage;
	ITMShortImage* inputRawDepthImage;
	const ORUtils::SE3Pose* inputPose;
	ITMLib::ITMIMUMeasurement* inputIMUMeasurement;

	StopWatchInterface* timer_instant;
	StopWatchInterface* timer_average;

	char* outFolder;

	int currentFrameNo;

//	void CollectICPErrorImages();
	std::vector<ORUtils::SE3Pose> trackingPoses;

	inline void ComputeICPErrors()
	{
		if (trackingPoses.empty())
			return;

		ITMLib::ITMView* view = nullptr;
		ITMViewBuilder* viewBuilder = ITMViewBuilderFactory::MakeViewBuilder(appData->imageSource->getCalib(),

		                                                                     appData->internalSettings->deviceType);
		// needs to be called once with view=nullptr for initialization
		viewBuilder->UpdateView(&view, inputRGBImage, inputRawDepthImage,
		                        appData->internalSettings->useDepthFilter, appData->internalSettings->useBilateralFilter);

		view = mainEngine->GetView();

		char str[250];
		sprintf(str, "%s/icp_error.txt", outFolder);
		std::ofstream icp_file(str);
		icp_file << "# MAE RMSE ICP_MAE ICP_RMSE" << std::endl;

		sprintf(str, "%s/photometric_error.txt", outFolder);
		std::ofstream photometric_file(str);
		photometric_file << "# MAE RMSE ICP_MAE ICP_RMSE" << std::endl;

		printf("Compute ICP Error: ");
		auto* rgbImage = new ITMUChar4Image(true, false);
		auto* depthImage = new ITMShortImage(true, false);
		int lastPercentile = -1;
		int count = 0;
		appData->imageSource->reset(); // reset to first image, re-iterate
		while (appData->imageSource->hasMoreImages() and count < trackingPoses.size())
		{

			int percentile = (count * 10) / trackingPoses.size();
			if (percentile != lastPercentile)
			{
				printf("%i%%\t", percentile * 10);
				lastPercentile = percentile;
			}
			appData->imageSource->getImages(rgbImage, depthImage);

			viewBuilder->UpdateView(&view, rgbImage, depthImage,
			                        appData->internalSettings->useDepthFilter,
			                        appData->internalSettings->useBilateralFilter);

			ORUtils::SE3Pose* pose = &trackingPoses.at(count);
			mainEngine->GetTrackingState()->pose_d->SetFrom(pose);

			ITMRenderError result = mainEngine->ComputeICPError();
			icp_file << result.MAE << " " << result.RMSE << " " << result.icpMAE << " " << result.icpRMSE << std::endl;

			ITMRenderError resultPhotometric = mainEngine->ComputePhotometricError();
			photometric_file << resultPhotometric.MAE << " " << resultPhotometric.RMSE << " " << resultPhotometric.icpMAE
			                 << " " << resultPhotometric.icpRMSE << std::endl;
			count += 1;
		}
		icp_file.close();
		photometric_file.close();

		free(rgbImage);
		free(depthImage);
		free(viewBuilder);
	}

	inline void CollectICPErrorImages()
	{
		if (trackingPoses.empty())
			return;

		ITMLib::ITMView* view = nullptr;
		ITMViewBuilder* viewBuilder = ITMViewBuilderFactory::MakeViewBuilder(appData->imageSource->getCalib(),
		                                                                     appData->internalSettings->deviceType);
		// needs to be called once with view=nullptr for initialization
		viewBuilder->UpdateView(&view, inputRGBImage, inputRawDepthImage,
		                        appData->internalSettings->useDepthFilter,
		                        appData->internalSettings->useBilateralFilter);
		view = mainEngine->GetView();

		ITMUChar4Image* outputImage = new ITMUChar4Image(inputRGBImage->noDims, true, false);
		char str[250];

		auto* rgbImage = new ITMUChar4Image(true, false);
		auto* depthImage = new ITMShortImage(true, false);
		int lastPercentile = -1;
		int count = 0;
		appData->imageSource->reset(); // reset to first image, re-iterate
		while (appData->imageSource->hasMoreImages() and count < trackingPoses.size())
		{
			int percentile = (count * 10) / trackingPoses.size();
			if (percentile != lastPercentile)
			{
				printf("%i%%\t", percentile * 10);
				lastPercentile = percentile;
			}

			viewBuilder->UpdateView(&view, rgbImage, depthImage,
			                        appData->internalSettings->useDepthFilter,
			                        appData->internalSettings->useBilateralFilter);

			ORUtils::SE3Pose* pose = &trackingPoses.at(count);
			mainEngine->GetTrackingState()->pose_d->SetFrom(pose);

			mainEngine->GetImage(outputImage, ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_ICP_ERROR,
			                     pose, &view->calib.intrinsics_d, appData->internalSettings->useSDFNormals);

			sprintf(str, "%s/recording/error_%04zu.ppm", outFolder, count);
			SaveImageToFile(outputImage, str);
		}

		free(outputImage);
		free(viewBuilder);
	}

	/**
	 * Collect and store point cloud renderings from every N-th pose
	 * @param N
	 */
	inline void CollectPointClouds(int N = 1)
	{
		if (trackingPoses.empty())
			return;

		ITMLib::ITMView* view = nullptr;
		ITMViewBuilder* viewBuilder = ITMViewBuilderFactory::MakeViewBuilder(appData->imageSource->getCalib(),
		                                                                     appData->internalSettings->deviceType);
		// needs to be called once with view=nullptr for initialization
		viewBuilder->UpdateView(&view, inputRGBImage, inputRawDepthImage,
		                        appData->internalSettings->useDepthFilter,
		                        appData->internalSettings->useBilateralFilter);
		view = mainEngine->GetView();


		ITMUChar4Image* outputImage = new ITMUChar4Image(inputRGBImage->noDims, true, false);
		char str[250];

		ORUtils::Image<ORUtils::Vector4<float>>* points = new ORUtils::Image<ORUtils::Vector4<float>>(
			mainEngine->GetView()->calib.intrinsics_d.imgSize, true, true);
		ORUtils::Image<ORUtils::Vector4<float>>* normals = new ORUtils::Image<ORUtils::Vector4<float>>(
			mainEngine->GetView()->calib.intrinsics_d.imgSize, true, true);

		auto* rgbImage = new ITMUChar4Image(true, false);
		auto* depthImage = new ITMShortImage(true, false);
		int lastPercentile = -1;
		int count = 0;
		appData->imageSource->reset(); // reset to first image, re-iterate
		while (appData->imageSource->hasMoreImages() and count < trackingPoses.size())
		{
			int percentile = (count * 10) / trackingPoses.size();
			if (percentile != lastPercentile)
			{
				printf("%i%%\t", percentile * 10);
				lastPercentile = percentile;
			}

			viewBuilder->UpdateView(&view, rgbImage, depthImage,
			                        appData->internalSettings->useDepthFilter,
			                        appData->internalSettings->useBilateralFilter);

			ORUtils::SE3Pose* pose = &trackingPoses.at(count);
			mainEngine->GetTrackingState()->pose_d->SetFrom(pose);

			mainEngine->GetImage(outputImage, ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_ICP_ERROR,
			                     pose, &mainEngine->GetView()->calib.intrinsics_d, appData->internalSettings->useSDFNormals);


			if (appData->internalSettings->deviceType == ITMLibSettings::DEVICE_CUDA)
			{
				points->SetFrom(mainEngine->GetTrackingState()->pointCloud->locations,
				                ORUtils::CUDA_TO_CPU);
				normals->SetFrom(mainEngine->GetTrackingState()->pointCloud->locations,
				                 ORUtils::CUDA_TO_CPU);
			} else
			{
				points->SetFrom(mainEngine->GetTrackingState()->pointCloud->locations,
				                ORUtils::CPU_TO_CPU);
				normals->SetFrom(mainEngine->GetTrackingState()->pointCloud->locations,
				                 ORUtils::CPU_TO_CPU);
			}

			sprintf(str, "%s/cloud_%04zu.pcd", outFolder, count);
			SavePointCloudToPCL(
				points->GetData(MEMORYDEVICE_CPU), normals->GetData(MEMORYDEVICE_CPU),
				inputRawDepthImage->noDims, *pose, str);
		}

		free(points);
		free(normals);
		free(outputImage);
		free(viewBuilder);
	}

	virtual void _initialise(int argc, char** argv, AppData* appData, ITMMainEngine* mainEngine) = 0;

	virtual bool _processFrame() = 0;

public:
	AppEngine() = default;

	virtual ~AppEngine()
	{
		sdkDeleteTimer(&timer_instant);
		sdkDeleteTimer(&timer_average);

		delete inputRGBImage;
		delete inputRawDepthImage;
		delete inputIMUMeasurement;
	}

	inline bool ProcessFrame()
	{
		if (!appData->imageSource->hasMoreImages()) return false;
		appData->imageSource->getImages(inputRGBImage, inputRawDepthImage);

		if (appData->imuSource != nullptr)
		{
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

		sdkResetTimer(&timer_instant);
		sdkStartTimer(&timer_instant);
		sdkStartTimer(&timer_average);

		if (not _processFrame()) return false;

		sdkStopTimer(&timer_instant);
		sdkStopTimer(&timer_average);

		trackingPoses.push_back(*mainEngine->GetTrackingState()->pose_d);
		statisticsEngine.LogTimeStats(mainEngine->GetTimeStats());
		statisticsEngine.LogPose(*mainEngine->GetTrackingState());
		statisticsEngine.LogBlockAllocations(mainEngine->GetAllocationsPerDirection());

		currentFrameNo++;

		return true;
	}

	inline void Initialise(int argc, char** argv, AppData* appData, ITMMainEngine* mainEngine)
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
