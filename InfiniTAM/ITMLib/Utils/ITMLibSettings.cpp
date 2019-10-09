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
	else if (iequals(type, "metal"))
		return ITMLibSettings::DeviceType::DEVICE_METAL;

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

ITMLibSettings::LibMode LibModeFromString(const std::string& mode, const bool useLoopClosure)
{
	if (iequals(mode, "voxel"))
	{
		if (useLoopClosure)
			return ITMLibSettings::LIBMODE_LOOPCLOSURE;
		else
			return ITMLibSettings::LIBMODE_BASIC;
	} else if (iequals(mode, "surfel"))
		return ITMLibSettings::LIBMODE_LOOPCLOSURE;

	printf(R"(ERROR: Unknown lib mode "%s". Using "voxel" instead)", mode.c_str());
	if (useLoopClosure)
		return ITMLibSettings::LIBMODE_LOOPCLOSURE;
	else
		return ITMLibSettings::LIBMODE_BASIC;
}

ITMLibSettings::ITMLibSettings()
	: fusionParams(),
	  sceneParams(0.04f, 5000, 0.005f, 0.2f, 6.0f, false),
	  surfelSceneParams(0.5f, 0.6f, static_cast<float>(20 * M_PI / 180), 0.01f, 0.004f, 3.5f, 25.0f, 4, 1.0f, 5.0f, 20,
	                    10000000, true, true)
{
	// skips every other point when using the colour renderer for creating a point cloud
	skipPoints = false;

	// create all the things required for marching cubes and mesh extraction
	// - uses additional memory (lots!)
	createMeshingEngine = false;

#ifndef COMPILE_WITHOUT_CUDA
	deviceType = DEVICE_CUDA;
#else
#ifdef COMPILE_WITH_METAL
	deviceType = DEVICE_METAL;
#else
	deviceType = DEVICE_CPU;
#endif
#endif

//	deviceType = DEVICE_CPU;

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
//	libMode = LIBMODE_BASIC_SURFELS;

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

	// Surfel tracking
	if (libMode == LIBMODE_BASIC_SURFELS)
	{
		trackerConfig = "extended,levels=rrbb,minstep=1e-4,outlierSpaceC=0.1,outlierSpaceF=0.004,numiterC=20,numiterF=20,tukeyCutOff=8,framesToSkip=0,framesToWeight=1,failureDec=20.0";
	}
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
	libMode = LibModeFromString(root["libMode"].as<std::string>(), useLoopClosure);

	if (libMode == LIBMODE_BASIC_SURFELS)
		trackerConfig = root["surfelTrackerConfig"].as<std::string>();
	else
		trackerConfig = root["voxelTrackerConfig"].as<std::string>();

	createMeshingEngine = root["createMeshingEngine"].as<bool>();
	swappingMode = SwappingModeFromString(root["swappingMode"].as<std::string>());
	behaviourOnFailure = FailureModeFromString(root["behaviourOnFailure"].as<std::string>());
	useBilateralFilter = root["useBilateralFilter"].as<bool>();
	skipPoints = root["skipPoints"].as<bool>();
	useApproximateRaycast = root["useApproximateRaycast"].as<bool>();

	const YAML::Node& voxelParamsNode = root["voxelSceneParams"];
	if (voxelParamsNode["voxelSize"].IsDefined())
		sceneParams.voxelSize = voxelParamsNode["voxelSize"].as<float>();
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


	const YAML::Node& surfelParamsNode = root["surfelSceneParams"];
	if (surfelParamsNode["deltaRadius"].IsDefined())
		surfelSceneParams.deltaRadius = surfelParamsNode["deltaRadius"].as<float>();
	if (surfelParamsNode["gaussianConfidenceSigma"].IsDefined())
		surfelSceneParams.gaussianConfidenceSigma = surfelParamsNode["gaussianConfidenceSigma"].as<float>();
	if (surfelParamsNode["maxMergeAngle"].IsDefined())
		surfelSceneParams.maxMergeAngle = surfelParamsNode["maxMergeAngle"].as<float>();
	if (surfelParamsNode["maxMergeDist"].IsDefined())
		surfelSceneParams.maxMergeDist = surfelParamsNode["maxMergeDist"].as<float>();
	if (surfelParamsNode["maxSurfelRadius"].IsDefined())
		surfelSceneParams.maxSurfelRadius = surfelParamsNode["maxSurfelRadius"].as<float>();
	if (surfelParamsNode["minRadiusOverlapFactor"].IsDefined())
		surfelSceneParams.minRadiusOverlapFactor = surfelParamsNode["minRadiusOverlapFactor"].as<float>();
	if (surfelParamsNode["stableSurfelConfidence"].IsDefined())
		surfelSceneParams.stableSurfelConfidence = surfelParamsNode["stableSurfelConfidence"].as<float>();
	if (surfelParamsNode["supersamplingFactor"].IsDefined())
		surfelSceneParams.supersamplingFactor = surfelParamsNode["supersamplingFactor"].as<int>();
	if (surfelParamsNode["trackingSurfelMaxDepth"].IsDefined())
		surfelSceneParams.trackingSurfelMaxDepth = surfelParamsNode["trackingSurfelMaxDepth"].as<float>();
	if (surfelParamsNode["trackingSurfelMinConfidence"].IsDefined())
		surfelSceneParams.trackingSurfelMinConfidence = surfelParamsNode["trackingSurfelMinConfidence"].as<float>();
	if (surfelParamsNode["unstableSurfelPeriod"].IsDefined())
		surfelSceneParams.unstableSurfelPeriod = surfelParamsNode["unstableSurfelPeriod"].as<int>();
	if (surfelParamsNode["unstableSurfelZOffset"].IsDefined())
		surfelSceneParams.unstableSurfelZOffset = surfelParamsNode["unstableSurfelZOffset"].as<int>();
	if (surfelParamsNode["useGaussianSampleConfidence"].IsDefined())
		surfelSceneParams.useGaussianSampleConfidence = surfelParamsNode["useGaussianSampleConfidence"].as<bool>();
	if (surfelParamsNode["useSurfelMerging"].IsDefined())
		surfelSceneParams.useSurfelMerging = surfelParamsNode["useSurfelMerging"].as<bool>();


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