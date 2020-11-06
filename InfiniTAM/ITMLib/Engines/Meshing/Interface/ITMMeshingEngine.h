// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <math.h>

#include "../../../Objects/Meshing/ITMMesh.h"
#include "../../../Objects/Scene/ITMScene.h"

namespace ITMLib
{
	class ITMMeshingEngine
	{
	public:
		virtual void MeshScene(ITMMesh *mesh, const Scene *scene) = 0;

		ITMMeshingEngine(void) { }
		virtual ~ITMMeshingEngine(void) { }
	};
}
