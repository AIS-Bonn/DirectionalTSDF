#include "ITMScene.h"
#include <ITMLib/ITMLibDefines.h>

#include "TSDF_CPU.h"
#include "TSDF_CUDA.h"

namespace ITMLib
{

template<class TVoxel>
ITMScene<TVoxel>::ITMScene(const ITMSceneParams* _sceneParams, bool _useSwapping, bool _directional,
                           MemoryDeviceType _memoryType)
	: sceneParams(_sceneParams)
{
	if (_directional)
	{
		if (_memoryType == MEMORYDEVICE_CPU)
			tsdfDirectional = new TSDF_CPU<ITMIndexDirectional, TVoxel>(sceneParams->allocationSize);
		else
			tsdfDirectional = new TSDF_CUDA<ITMIndexDirectional, TVoxel>(sceneParams->allocationSize);

	} else
	{
		if (_memoryType == MEMORYDEVICE_CPU)
			tsdf = new TSDF_CPU<ITMIndex, TVoxel>(sceneParams->allocationSize);
		else
			tsdf = new TSDF_CUDA<ITMIndex, TVoxel>(sceneParams->allocationSize);

	}
}

template
class ITMScene<ITMVoxel>;

} // namespace ITMLib