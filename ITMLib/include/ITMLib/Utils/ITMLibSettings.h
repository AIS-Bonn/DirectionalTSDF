// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <string>
#include <cmath>

#include <ITMLib/Utils/ITMFusionParams.h>
#include <ITMLib/Utils/ITMSceneParams.h>
#include <ORUtils/MemoryDeviceType.h>

namespace ITMLib
{

class ITMLibSettings
{
public:
	/// The device used to run the DeviceAgnostic code
	typedef enum
	{
		DEVICE_CPU,
		DEVICE_CUDA,
	} DeviceType;

	typedef enum
	{
		FAILUREMODE_RELOCALISE,
		FAILUREMODE_IGNORE,
		FAILUREMODE_STOP_INTEGRATION
	} FailureMode;

	typedef enum
	{
		SWAPPINGMODE_DISABLED,
		SWAPPINGMODE_ENABLED,
		SWAPPINGMODE_DELETE
	} SwappingMode;

	typedef enum
	{
		LIBMODE_BASIC,
		LIBMODE_LOOPCLOSURE
	} LibMode;

	/// Select the type of device to use
	DeviceType deviceType;

	bool useApproximateRaycast;

	/// Whether to apply pre-filter do depth image
	bool useDepthFilter;

	/// Whether to bilateral filter do depth image
	bool useBilateralFilter;

	/// Whether to infer normals from SDF (slower) or neighboring points for tracking
	bool useSDFNormals;

	/// For ITMColorTracker: skip every other point in energy function evaluation.
	bool skipPoints;

	bool createMeshingEngine;

	FailureMode behaviourOnFailure;
	SwappingMode swappingMode;
	LibMode libMode;

	std::string trackerConfig;

	ITMFusionParams fusionParams;

	/// Further, scene specific parameters such as voxel size
	ITMSceneParams sceneParams;

	ITMLibSettings();

	virtual ~ITMLibSettings() = default;

	explicit ITMLibSettings(const std::string& settingsFile);

	// Suppress the default copy constructor and assignment operator
	ITMLibSettings(const ITMLibSettings&);

	ITMLibSettings& operator=(const ITMLibSettings&);

	[[nodiscard]] MemoryDeviceType GetMemoryType() const;

	[[nodiscard]] bool Directional() const;
};
}
