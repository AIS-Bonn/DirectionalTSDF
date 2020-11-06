// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMMeshingEngine.h"

namespace ITMLib
{
class ITMMeshingEngine_CPU : public ITMMeshingEngine
{
public:
	void MeshScene(ITMMesh* mesh, const Scene* scene);

	ITMMeshingEngine_CPU(void)
	{}

	~ITMMeshingEngine_CPU(void)
	{}
};
}
