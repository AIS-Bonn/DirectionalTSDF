// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <iostream>
#include <ITMLib/Objects/Scene/TSDF.h>
#include <ITMLib/Utils/ITMSceneParams.h>

namespace ITMLib
{
/** \brief
Represents the 3D world model as a hash of small voxel
blocks
*/
template<class TVoxel>
class ITMScene
{
public:
	/** Scene parameters like voxel size etc. */
	const ITMSceneParams* sceneParams = nullptr;

	TSDF<IndexShort, TVoxel>* tsdf = nullptr;

	TSDF<IndexDirectionalShort, TVoxel>* tsdfDirectional = nullptr;

	void SaveToDirectory(const std::string& outputDirectory) const
	{
		std::cerr << "SaveToDirectory not implemented" << std::endl;
	}

	void LoadFromDirectory(const std::string& outputDirectory)
	{
		std::cerr << "LoadFromDirectory not implemented" << std::endl;
	}

	void Clear()
	{
		if (tsdf)
			tsdf->clear();
		if (tsdfDirectional)
			tsdfDirectional->clear();
	}

	ITMScene(const ITMSceneParams* _sceneParams, bool _useSwapping, bool _directional, MemoryDeviceType _memoryType);

	~ITMScene() = default;

	// Suppress the default copy constructor and assignment operator
	ITMScene(const ITMScene&) = delete;

	ITMScene& operator=(const ITMScene&) = delete;
};
}
