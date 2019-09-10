// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMLibSettings.h"
using namespace ITMLib;

#include <climits>
#include <cmath>

ITMLibSettings::ITMLibSettings(void)
//:	sceneParams(0.04f, 5000, 0.005f, 0.2f, 3.0f, false),
:	sceneParams(0.04f, 50000, 0.01f, 0.2f, 3.0f, false),
//:	sceneParams(0.08f, 10000, 0.02f, 0.2f, 6.0f, false),
	surfelSceneParams(0.5f, 0.6f, static_cast<float>(20 * M_PI / 180), 0.01f, 0.004f, 3.5f, 25.0f, 4, 1.0f, 5.0f, 20, 10000000, true, true)
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

//	fusionParams.tsdfMode = TSDFMODE_DEFAULT;
	fusionParams.tsdfMode = TSDFMODE_DIRECTIONAL;

//	fusionParams.fusionMode = FUSIONMODE_VOXEL_PROJECTION;
//	fusionParams.fusionMode = FUSIONMODE_RAY_CASTING_NORMAL;
	fusionParams.fusionMode = FUSIONMODE_RAY_CASTING_VIEW_DIR_AND_NORMAL;
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
	if(libMode == LIBMODE_BASIC_SURFELS)
	{
		trackerConfig = "extended,levels=rrbb,minstep=1e-4,outlierSpaceC=0.1,outlierSpaceF=0.004,numiterC=20,numiterF=20,tukeyCutOff=8,framesToSkip=0,framesToWeight=1,failureDec=20.0";
	}
}

MemoryDeviceType ITMLibSettings::GetMemoryType() const
{
	return deviceType == ITMLibSettings::DEVICE_CUDA ? MEMORYDEVICE_CUDA : MEMORYDEVICE_CPU;
}
