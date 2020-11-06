// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMMeshingEngine.h"

namespace ITMLib
{
class ITMMeshingEngine_CUDA : public ITMMeshingEngine
{
private:
	unsigned int* noTriangles_device;
	Vector4s* visibleBlockGlobalPos_device;

public:
	void MeshScene(ITMMesh* mesh, const Scene* scene);

	ITMMeshingEngine_CUDA(void);

	~ITMMeshingEngine_CUDA(void);
};
}
