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
	ORUtils::SE3Pose initialPose;
	int numberExportPointClouds;
	int maxNumFrames;
	bool computePostError;

	AppData()
		: imageSource(nullptr), imuSource(nullptr), trajectorySource(nullptr),
		  internalSettings(nullptr), outputDirectory("./Output"), numberExportPointClouds(-1), maxNumFrames(-1),
		  computePostError(false)
	{}
};

inline int ParseCLIOptions(int argc, char** argv,
                           AppData& appData)
{

	CLI::App app{"RGB-D reconstruction with visualization"};

	std::string calibrationFile, settingsFile, datasetDirectory, trajectoryFile;
	std::vector<std::string> device, rawPaths, videoPaths;
	std::vector<float> initialPose;
	std::vector<int> numberExportedPointClouds;

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
	                                "Use raw dataset (e.g. rgb/%04i.ppm depth/%04i.pgm imu/%04i.txt)")
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

	auto initialPoseOption = app.add_option("--initial_pose", initialPose, "Start pose for tracking")
		->type_name("x y z rx ry rz rw")
		->excludes(trajectoryOption)
		->expected(7);

	app.add_option("--export_point_clouds", appData.numberExportPointClouds,
	               "Export rendered point clouds from tracking poses after finishing. N number total number of point clouds (evenly spaced). Default (0) = every pose.")
		->type_name("N")
		->default_str("0");

	app.add_option("--max_frames", appData.maxNumFrames,
	               "Maximum number of frames to process. After that, stop. Default (-1) = no limit")
		->type_name("N")
		->default_str("-1");

	app.add_flag("--compute_post_error", appData.computePostError,
	             "Re-render all views from tracked poses and compare to input data");

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

	if (initialPoseOption->count())
	{
		appData.initialPose.SetFrom(initialPose.at(0), initialPose.at(1), initialPose.at(2),
		                            initialPose.at(3), initialPose.at(4), initialPose.at(5), initialPose.at(6));
	}

	if (settingsFile.empty())
		appData.internalSettings = std::make_shared<ITMLibSettings>();
	else
		appData.internalSettings = std::make_shared<ITMLibSettings>(settingsFile);

	return 0;
}
