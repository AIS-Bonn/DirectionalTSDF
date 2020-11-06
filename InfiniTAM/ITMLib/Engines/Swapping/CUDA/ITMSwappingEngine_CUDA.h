// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../Interface/ITMSwappingEngine.h"

namespace ITMLib
{
template<class TVoxel>
class ITMSwappingEngine_CUDA : public ITMSwappingEngine<TVoxel>
{
private:
	int* noNeededEntries_device, * noAllocatedVoxelEntries_device;
	int* entriesToClean_device;

	int LoadFromGlobalMemory(ITMScene<TVoxel>* scene);

public:
	void IntegrateGlobalIntoLocal(ITMScene<TVoxel>* scene, ITMRenderState* renderState);

	void SaveToGlobalMemory(ITMScene<TVoxel>* scene, ITMRenderState* renderState);

	void CleanLocalMemory(ITMScene<TVoxel>* scene, ITMRenderState* renderState);

	ITMSwappingEngine_CUDA(void);

	~ITMSwappingEngine_CUDA(void);
};
}
