// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMRenderState_VH.h"
#include "../../Utils/ITMSceneParams.h"

namespace ITMLib
{
struct ITMRenderStateFactory
{
	/** Creates a render state, containing rendering info for the scene. */
	static ITMRenderState*
	CreateRenderState(const Vector2i& imgSize, const ITMSceneParams* sceneParams, MemoryDeviceType memoryType)
	{
		return new ITMRenderState_VH(ITMVoxelBlockHash::noTotalEntries, imgSize, sceneParams->viewFrustum_min,
		                             sceneParams->viewFrustum_max, memoryType);
	}
};
}
