#include "ITMScene.h"
#include <ITMLib/ITMLibDefines.h>

#include "TSDF_CPU.h"
#include "TSDF_CUDA.h"

namespace ITMLib
{

template<class TVoxel>
ITMScene<TVoxel>::ITMScene(const ITMSceneParams* _sceneParams, bool _useSwapping, MemoryDeviceType _memoryType)
	: sceneParams(_sceneParams)
	, index(_memoryType),
	  localVBA(_memoryType, 1, 1)//index.getNumAllocatedVoxelBlocks(), index.getVoxelBlockSize())
{
	index.Reset();
	if (_useSwapping) globalCache = new ITMGlobalCache<TVoxel>();
	else globalCache = NULL;

	if (_memoryType == MEMORYDEVICE_CPU)
		tsdf = new TSDF_CPU<IndexDirectionalShort, TVoxel>(2e5);
	else
		tsdf = new TSDF_CUDA<IndexDirectionalShort, TVoxel>(2e5);
}

template class ITMScene<ITMVoxel>;

} // namespace ITMLib