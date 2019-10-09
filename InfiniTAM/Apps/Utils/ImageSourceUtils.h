//
// Created by Malte Splietker on 02.10.19.
//

#pragma once

#include "ITMLib/Utils/ITMLibSettings.h"
#include "InputSource/ImageSourceEngine.h"

using namespace InputSource;
using namespace ITMLib;

/**
 * Create Image source for the specified device (or automatically choose available device)
 * @param imageSource
 * @param imuSource
 * @param calibrationFile
 * @param deviceType Device Type (auto, viewer, OpenNI, UVC, Realsense, Kinect2, PicoFlexx)
 * @param deviceId Optional device ID
 */
static inline void CreateDeviceImageSource(ImageSourceEngine*& imageSource, IMUSourceEngine*& imuSource,
                                           const std::string& calibrationFile, const std::string& deviceType,
                                           const std::string& deviceId)
{
	if (iequals(deviceType, "viewer"))
	{
		imageSource = new BlankImageGenerator("", Vector2i(640, 480));
		return;
	}

	if (iequals(deviceType, "OpenNI") or (iequals(deviceType, "auto") and not imageSource))
	{
		// If no calibration file specified, use the factory default calibration
		bool useInternalCalibration = calibrationFile.empty();

		printf("trying OpenNI device: %s - calibration: %s\n",
		       (not deviceId.empty()) ? deviceId.c_str() : "<OpenNI default device>",
		       useInternalCalibration ? "internal" : "from file");
		imageSource = new OpenNIEngine(calibrationFile.c_str(),
		                               deviceId.empty() ? nullptr : deviceId.c_str(),
		                               useInternalCalibration);
		if (imageSource->getDepthImageSize().x == 0)
		{
			delete imageSource;
			imageSource = nullptr;
		}
	}
	if (iequals(deviceType, "UVC") or (iequals(deviceType, "auto") and not imageSource))
	{
		printf("trying UVC device\n");
		imageSource = new LibUVCEngine(calibrationFile.c_str());
		if (imageSource->getDepthImageSize().x == 0)
		{
			delete imageSource;
			imageSource = nullptr;
		}
	}
	if (iequals(deviceType, "Realsense") or (iequals(deviceType, "auto") and not imageSource))
	{
		printf("trying RealSense device with SDK 2.X (librealsense2)\n");
		imageSource = new RealSense2Engine(calibrationFile.c_str());
		if (imageSource->getDepthImageSize().x == 0)
		{
			delete imageSource;
			imageSource = nullptr;
		}
	}
	if (iequals(deviceType, "Kinect2") or (iequals(deviceType, "auto") and not imageSource))
	{
		printf("trying MS Kinect 2 device\n");
		imageSource = new Kinect2Engine(calibrationFile.c_str());
		if (imageSource->getDepthImageSize().x == 0)
		{
			delete imageSource;
			imageSource = nullptr;
		}
	}
	if (iequals(deviceType, "PicoFlexx") or (iequals(deviceType, "auto") and not imageSource))
	{
		printf("trying PMD PicoFlexx device\n");
		imageSource = new PicoFlexxEngine(calibrationFile.c_str());
		if (imageSource->getDepthImageSize().x == 0)
		{
			delete imageSource;
			imageSource = nullptr;
		}
	}
}

/**
 * Create image source for raw images given by file path mask (e.g. /foo/bar/file%%04i.png)
 * @param imageSource
 * @param imuSource
 * @param calibrationFile
 * @param rgbMask
 * @param depthMask
 * @param imuMask
 */
static inline void CreateRAWImageSource(ImageSourceEngine*& imageSource, IMUSourceEngine*& imuSource,
                                        const std::string& calibrationFile, const std::string& rgbMask,
                                        const std::string& depthMask, const std::string& imuMask)
{
	if (imuMask.empty())
	{
		ImageMaskPathGenerator pathGenerator(rgbMask.c_str(), depthMask.c_str());
		imageSource = new ImageFileReader<ImageMaskPathGenerator>(calibrationFile.c_str(), pathGenerator);
	} else
	{
		imageSource = new RawFileReader(calibrationFile.c_str(), rgbMask.c_str(), depthMask.c_str(), Vector2i(320, 240),
		                                0.5f);
		imuSource = new IMUSourceEngine(imuMask.c_str());
	}

	if (imageSource->getDepthImageSize().x == 0)
	{
		delete imageSource;
		delete imuSource;
		imuSource = nullptr;
		imageSource = nullptr;
	}
}

/**
 * Create image source for video file input
 * @param imageSource
 * @param imuSource
 * @param calibrationFile
 * @param rgbFile
 * @param depthFile
 */
static inline void CreateFFMPEGImageSource(ImageSourceEngine*& imageSource, IMUSourceEngine*& imuSource,
                                           const std::string& calibrationFile, const std::string& rgbFile,
                                           const std::string& depthFile)
{
	imageSource = new InputSource::FFMPEGReader(calibrationFile.c_str(), rgbFile.c_str(), depthFile.c_str());
	if (imageSource->getDepthImageSize().x == 0)
	{
		delete imageSource;
		imageSource = nullptr;
	}
}

/**
 * Create image source for TUM-style datasets
 * @param imageSource
 * @param imuSource
 * @param calibrationFile
 * @param datasetDirectory
 */
static inline void CreateTUMImageSource(ImageSourceEngine*& imageSource, IMUSourceEngine*& imuSource,
                                        const std::string& calibrationFile, const std::string& datasetDirectory)
{
	TUMPathGenerator pathGenerator(datasetDirectory);
	imageSource = new ImageFileReader<TUMPathGenerator>(calibrationFile.c_str(), pathGenerator);
	imuSource = nullptr;
}

