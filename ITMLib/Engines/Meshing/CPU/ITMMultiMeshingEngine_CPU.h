// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMMultiMeshingEngine.h"
#include "../../../Objects/Scene/ITMMultiSceneAccess.h"

namespace ITMLib
{
class ITMMultiMeshingEngine_CPU : public ITMMultiMeshingEngine
{
	void MeshScene(ITMMesh* mesh, const MultiSceneManager& sceneManager);
};
}