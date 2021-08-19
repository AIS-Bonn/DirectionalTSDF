// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <ITMLib/Objects/Scene/ITMMultiSceneAccess.h>
#include "ITMMeshingEngine.h"
#include "../../../Objects/Meshing/ITMMesh.h"
#include "../../MultiScene/ITMMapGraphManager.h"

namespace ITMLib
{
	class ITMMultiMeshingEngine
	{
	public:
		typedef typename ITMMultiIndex<ITMVoxelIndex>::IndexData MultiIndexData;
		typedef ITMMultiVoxel<ITMVoxel> MultiVoxelData;
		typedef ITMVoxelMapGraphManager<ITMVoxel> MultiSceneManager;

		virtual ~ITMMultiMeshingEngine(void) {}

		virtual void MeshScene(ITMMesh *mesh, const ITMVoxelMapGraphManager<ITMVoxel> & sceneManager) = 0;
	};
}