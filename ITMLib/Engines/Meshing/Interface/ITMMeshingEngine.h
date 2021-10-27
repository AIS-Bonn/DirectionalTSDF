// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <ITMLib/Objects/Meshing/ITMMesh.h>
#include <ITMLib/Objects/Scene/ITMScene.h>
#include <ITMLib/ITMLibDefines.h>

namespace ITMLib
{
class ITMMeshingEngine
{
public:
	virtual void MeshScene(ITMMesh* mesh, const Scene* scene) = 0;

	ITMMeshingEngine() = default;

	virtual ~ITMMeshingEngine() = default;
};
}