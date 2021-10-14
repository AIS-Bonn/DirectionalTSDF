// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMLibSettings.h"


#include <climits>
#include <cmath>

#include <yaml-cpp/yaml.h>

namespace ITMLib
{

ITMLibSettings::DeviceType DeviceTypeFromString(const std::string& type)
{
	if (iequals(type, "cpu"))
		return ITMLibSettings::DeviceType::DEVICE_CPU;
	else if (iequals(type, "cuda"))
		return ITMLibSettings::DeviceType::DEVICE_CUDA;

	printf(R"(ERROR: Unknown device type "%s". Using "cpu" instead)", type.c_str());
	return ITMLibSettings::DeviceType::DEVICE_CPU;
}

ITMLibSettings::FailureMode FailureModeFromString(const std::string& mode)
{
	if (iequals(mode, "relocalise"))
		return ITMLibSettings::FAILUREMODE_RELOCALISE;
	else if (iequals(mode, "ignore"))
		return ITMLibSettings::FAILUREMODE_IGNORE;
	else if (iequals(mode, "stopIntegration"))
		return ITMLibSettings::FAILUREMODE_STOP_INTEGRATION;

	printf(R"(ERROR: Unknown failure mode "%s". Using "relocalise" instead)", mode.c_str());
	return ITMLibSettings::FAILUREMODE_RELOCALISE;
}

ITMLibSettings::SwappingMode SwappingModeFromString(const std::string& mode)
{
	if (iequals(mode, "disabled"))
		return ITMLibSettings::SWAPPINGMODE_DISABLED;
	else if (iequals(mode, "enabled"))
		return ITMLibSettings::SWAPPINGMODE_ENABLED;
	else if (iequals(mode, "delete"))
		return ITMLibSettings::SWAPPINGMODE_DELETE;

	printf(R"(ERROR: Unknown swapping mode "%s". Using "disabled" instead)", mode.c_str());
	return ITMLibSettings::SWAPPINGMODE_DISABLED;
}

ITMLibSettings::LibMode getLibMode(const bool useLoopClosure)
{
	if (useLoopClosure)
		return ITMLibSettings::LIBMODE_LOOPCLOSURE;
	else
		return ITMLibSettings::LIBMODE_BASIC;
}

ITMLibSettings::ITMLibSettings()
	: fusionParams(),
	  sceneParams(0.04f, 5000, 0.005f, 0.2f, 6.0f, false)
{
	// skips every other point when using the colour renderer for creating a point cloud
	skipPoints = false;

	// create all the things required for marching cubes and mesh extraction
	// - uses additional memory (lots!)
	createMeshingEngine = false;

#ifndef COMPILE_WITHOUT_CUDA
	deviceType = DEVICE_CUDA;
#else
	deviceType = DEVICE_CPU;
#endif

	/// how swapping works: disabled, fully enabled (still with dragons) and delete what's not visible - not supported in loop closure version
	swappingMode = SWAPPINGMODE_DISABLED;

	/// enables or disables approximate raycast
	useApproximateRaycast = false;

	/// enable or disable bilateral depth filtering
	useBilateralFilter = false;

	/// what to do on tracker failure: ignore, relocalise or stop integration - not supported in loop closure version
	behaviourOnFailure = FAILUREMODE_RELOCALISE;

	/// switch between various library modes - basic, with loop closure, etc.
	libMode = LIBMODE_BASIC;
//	libMode = LIBMODE_LOOPCLOSURE;

	//////////////////////////////////////////////////////////////////////////
	/// Fusion Params
	//////////////////////////////////////////////////////////////////////////

	fusionParams.tsdfMode = TSDFMODE_DEFAULT;
//	fusionParams.tsdfMode = TSDFMODE_DIRECTIONAL;

	fusionParams.fusionMode = FUSIONMODE_VOXEL_PROJECTION;
//	fusionParams.fusionMode = FUSIONMODE_RAY_CASTING_NORMAL;
//	fusionParams.fusionMode = FUSIONMODE_RAY_CASTING_VIEW_DIR_AND_NORMAL;
//	fusionParams.fusionMode = FUSIONMODE_RAY_CASTING_VIEW_DIR;

	fusionParams.carvingMode = CARVINGMODE_VOXEL_PROJECTION;
//	fusionParams.carvingMode = CARVINGMODE_RAY_CASTING;

//	fusionParams.fusionMetric = FUSIONMETRIC_POINT_TO_POINT;
	fusionParams.fusionMetric = FUSIONMETRIC_POINT_TO_PLANE;

	fusionParams.useWeighting = true;

	fusionParams.useSpaceCarving = true;


	//////////////////////////////////////////////////////////////////////////
	/// Tracking Params
	//////////////////////////////////////////////////////////////////////////

	//// Default ICP tracking
//	trackerConfig = "type=icp,levels=rrrbb,minstep=1e-3,"
//					"outlierC=0.01,outlierF=0.002,"
//					"numiterC=10,numiterF=2,failureDec=5.0"; // 5 for normal, 20 for loop closure

	// Depth-only extended tracker:
	trackerConfig = "type=extended,levels=rrbb,useDepth=1,minstep=1e-4,"
	                "outlierSpaceC=0.1,outlierSpaceF=0.004,"
	                "numiterC=20,numiterF=50,tukeyCutOff=8,"
	                "framesToSkip=20,framesToWeight=50,failureDec=20.0";

	//// For hybrid intensity+depth tracking:
//	trackerConfig = "type=extended,levels=bbb,useDepth=1,useColour=1,"
//					  "colourWeight=0.3,minstep=1e-4,"
//					  "outlierColourC=0.175,outlierColourF=0.005,"
//					  "outlierSpaceC=0.1,outlierSpaceF=0.004,"
//					  "numiterC=20,numiterF=50,tukeyCutOff=8,"
//					  "framesToSkip=20,framesToWeight=50,failureDec=20.0";

	// Colour only tracking, using rendered colours
//	trackerConfig = "type=rgb,levels=rrbb";

	//trackerConfig = "type=imuicp,levels=tb,minstep=1e-3,outlierC=0.01,outlierF=0.005,numiterC=4,numiterF=2";
	//trackerConfig = "type=extendedimu,levels=ttb,minstep=5e-4,outlierSpaceC=0.1,outlierSpaceF=0.004,numiterC=20,numiterF=5,tukeyCutOff=8,framesToSkip=20,framesToWeight=50,failureDec=20.0";
}

MemoryDeviceType ITMLibSettings::GetMemoryType() const
{
	return deviceType == ITMLibSettings::DEVICE_CUDA ? MEMORYDEVICE_CUDA : MEMORYDEVICE_CPU;
}

ITMLibSettings::ITMLibSettings(const std::string& settingsFile)
	: ITMLibSettings()
{
	YAML::Node root = YAML::LoadFile(settingsFile);

	deviceType = DeviceTypeFromString(root["deviceType"].as<std::string>());

	bool useLoopClosure = false;
	if (root["useLoopClosure"].IsDefined())
		useLoopClosure = root["useLoopClosure"].as<bool>();
	libMode = getLibMode(useLoopClosure);

	trackerConfig = root["voxelTrackerConfig"].as<std::string>();

	createMeshingEngine = root["createMeshingEngine"].as<bool>();
	swappingMode = SwappingModeFromString(root["swappingMode"].as<std::string>());
	behaviourOnFailure = FailureModeFromString(root["behaviourOnFailure"].as<std::string>());
	useBilateralFilter = root["useBilateralFilter"].as<bool>();
	skipPoints = root["skipPoints"].as<bool>();
	useApproximateRaycast = root["useApproximateRaycast"].as<bool>();

	const YAML::Node& voxelParamsNode = root["voxelSceneParams"];
	if (voxelParamsNode["voxelSize"].IsDefined())
	{
		sceneParams.voxelSize = voxelParamsNode["voxelSize"].as<float>();
		sceneParams.oneOverVoxelSize = 1 / sceneParams.voxelSize;
	}
	if (voxelParamsNode["truncationDistance"].IsDefined())
		sceneParams.mu = voxelParamsNode["truncationDistance"].as<float>();
	if (voxelParamsNode["maxWeight"].IsDefined())
		sceneParams.maxW = voxelParamsNode["maxWeight"].as<int>();
	if (voxelParamsNode["stopIntegratingAtMaxWeight"].IsDefined())
		sceneParams.stopIntegratingAtMaxW = voxelParamsNode["stopIntegratingAtMaxWeight"].as<bool>();
	if (voxelParamsNode["minDistance"].IsDefined())
		sceneParams.viewFrustum_min = voxelParamsNode["minDistance"].as<float>();
	if (voxelParamsNode["maxDistance"].IsDefined())
		sceneParams.viewFrustum_max = voxelParamsNode["maxDistance"].as<float>();
	if (voxelParamsNode["allocationSize"].IsDefined())
		sceneParams.allocationSize = voxelParamsNode["allocationSize"].as<size_t>();


	const YAML::Node& fusionParamsNode = root["fusionParams"];
	if (fusionParamsNode["tsdfMode"].IsDefined())
		fusionParams.tsdfMode = TSDFModeFromString(fusionParamsNode["tsdfMode"].as<std::string>());
	if (fusionParamsNode["fusionMode"].IsDefined())
		fusionParams.fusionMode = FusionModeFromString(fusionParamsNode["fusionMode"].as<std::string>());
	if (fusionParamsNode["carvingMode"].IsDefined())
		fusionParams.carvingMode = CarvingModeFromString(fusionParamsNode["carvingMode"].as<std::string>());
	if (fusionParamsNode["fusionMetric"].IsDefined())
		fusionParams.fusionMetric = FusionMetricFromString(fusionParamsNode["fusionMetric"].as<std::string>());
	if (fusionParamsNode["useWeighting"].IsDefined())
		fusionParams.useWeighting = fusionParamsNode["useWeighting"].as<bool>();
	if (fusionParamsNode["useSpaceCarving"].IsDefined())
		fusionParams.useSpaceCarving = fusionParamsNode["useSpaceCarving"].as<bool>();
}

} // namespace ITMLib