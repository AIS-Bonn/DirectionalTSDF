//
// Created by Malte Splietker on 02.10.19.
//

#pragma once

#include <memory>
#include <string>

#include <CLI/CLI11.hpp>

#include "ITMLib/Utils/ITMLibSettings.h"
#include "InputSource/ImageSourceEngine.h"
#include "ImageSourceUtils.h"
#include "InputSource/TrajectorySourceEngine.h"

using namespace InputSource;
using namespace ITMLib;

struct AppData
{
	ImageSourceEngine* imageSource;
	IMUSourceEngine* imuSource;
	TrajectorySourceEngine* trajectorySource;
	std::shared_ptr<ITMLibSettings> internalSettings;
	std::string outputDirectory;

	AppData()
		: imageSource(nullptr), imuSource(nullptr), trajectorySource(nullptr),
		  internalSettings(nullptr), outputDirectory("./Output")
	{}
};

inline int ParseCLIOptions(int argc, char** argv,
                           AppData& appData)
{

	CLI::App app{"RGB-D reconstruction with visualization"};

	std::string calibrationFile, settingsFile, datasetDirectory, trajectoryFile;
	std::vector<std::string> device, rawPaths, videoPaths;

	app.add_option("-c,--calibration", calibrationFile,
	               "Path to the calibration file (required by all modes excluding Realsense)")
		->check(CLI::ExistingPath);
	app.add_option("-s,--settings", settingsFile,
	               "Path to the settings file")
		->check(CLI::ExistingPath);
	app.add_option("-o,--output", appData.outputDirectory,
	               "Directory to store all output in");
	auto deviceOption = app.add_option("--device", device,
	                                   "Device type")
		->check(CLI::IsMember({"auto", "viewer", "OpenNI", "UVC", "Realsense", "Kinect2", "PicoFlexx"}, CLI::ignore_case))
		->default_str("auto")
		->type_name("TYPE [DEVICE_ID]")
		->expected(-1);
	// Dataset options
	auto tumOption = app.add_option("--tum", datasetDirectory, "Use TUM style dataset")
		->excludes(deviceOption)
		->check(CLI::ExistingDirectory)
		->type_name("DIRECTORY");
	auto rawOption = app.add_option("--raw", rawPaths,
	                                "Use raw dataset (e.g. rgb/%%04i.ppm depth/%%04i.pgm imu/%%04i.txt)")
		->excludes(deviceOption, tumOption)
		->type_name("RGB DEPTH [IMU]")
		->expected(-2);
	auto videoOption = app.add_option("--video", videoPaths,
	                                  "Use video dataset (e.g. rgb.avi depth.avi)")
		->excludes(deviceOption, rawOption, tumOption)
		->type_name("RGB DEPTH")
		->expected(2);

	auto trajectoryOption = app.add_option("-t,--trajectory", trajectoryFile,
	                                       "Use trajectory file (TUM format) instead of ICP tracker");

	CLI11_PARSE(app, argc, argv)

	if (tumOption->count())
	{
		CreateTUMImageSource(appData.imageSource, appData.imuSource, calibrationFile, datasetDirectory);
	} else if (rawOption->count())
	{
		const std::string rgbMask = rawPaths.at(0);
		const std::string depthMask = rawPaths.at(1);
		std::string imuMask;
		if (rawPaths.size() > 2)
			imuMask = rawPaths.at(2);
		CreateRAWImageSource(appData.imageSource, appData.imuSource, calibrationFile, rgbMask, depthMask, imuMask);
		std::cout << rgbMask << ", " << depthMask << ", " << imuMask << "\n";
	} else if (videoOption->count())
	{
		const std::string rgbFile = videoPaths.at(0);
		const std::string depthFile = videoPaths.at(1);
		CreateFFMPEGImageSource(appData.imageSource, appData.imuSource, calibrationFile, rgbFile, depthFile);
	} else
	{ // Assume device mode
		std::string deviceType = "auto";
		std::string deviceId;
		if (not device.empty())
			deviceType = device.at(0);
		if (device.size() > 1)
			deviceId = device.at(1);
		CreateDeviceImageSource(appData.imageSource, appData.imuSource, calibrationFile, deviceType, deviceId);
	}

	// External trajectory provided
	if (trajectoryOption->count())
	{
		appData.trajectorySource = new TrajectorySourceEngine();
		appData.trajectorySource->Read(trajectoryFile);
	}

	if (settingsFile.empty())
		appData.internalSettings = std::make_shared<ITMLibSettings>();
	else
		appData.internalSettings = std::make_shared<ITMLibSettings>(settingsFile);

	return 0;
}
