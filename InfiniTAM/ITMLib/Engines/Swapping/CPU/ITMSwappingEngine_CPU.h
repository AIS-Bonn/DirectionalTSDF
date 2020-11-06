// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMSwappingEngine.h"

namespace ITMLib
{
template<class TVoxel>
class ITMSwappingEngine_CPU : public ITMSwappingEngine<TVoxel>
{
private:
	int LoadFromGlobalMemory(ITMScene<TVoxel>* scene);

public:
	// This class is currently just for debugging purposes -- swaps CPU memory to CPU memory.
	// Potentially this could stream into the host memory from somwhere else (disk, database, etc.).

	void IntegrateGlobalIntoLocal(ITMScene<TVoxel>* scene, ITMRenderState* renderState);

	void SaveToGlobalMemory(ITMScene<TVoxel>* scene, ITMRenderState* renderState);

	void CleanLocalMemory(ITMScene<TVoxel>* scene, ITMRenderState* renderState);

	ITMSwappingEngine_CPU(void);

	~ITMSwappingEngine_CPU(void);
};
}
