// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMMultiMeshingEngine.h"
#include "../../../Objects/Scene/ITMMultiSceneAccess.h"

namespace ITMLib
{
class ITMMultiMeshingEngine_CUDA : public ITMMultiMeshingEngine
{
private:
	unsigned int* noTriangles_device;
	Vector4s* visibleBlockGlobalPos_device;

public:
	MultiIndexData* indexData_device, indexData_host;
	MultiVoxelData* voxelData_device, voxelData_host;

	void MeshScene(ITMMesh* mesh, const MultiSceneManager& sceneManager);

	ITMMultiMeshingEngine_CUDA(void);

	~ITMMultiMeshingEngine_CUDA(void);
};
}

