// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "TrackerIterationType.h"
#include "../../Utils/ITMMath.h"
#include "../../../ORUtils/MemoryBlock.h"
#include "ITMHierarchyLevel.h"

namespace ITMLib
{
template<class ImageType>
class ITMTemplatedHierarchyLevel : public ITMHierarchyLevel
{
public:
	ImageType* data;

	ITMTemplatedHierarchyLevel(Vector2i imgSize, int levelId, TrackerIterationType iterationType,
	                           MemoryDeviceType memoryType, bool skipAllocation = false)
		: ITMHierarchyLevel(levelId, iterationType, skipAllocation)
	{
		if (!skipAllocation) this->data = new ImageType(imgSize, memoryType);
	}

	~ITMTemplatedHierarchyLevel()
	{
		if (manageData)
			delete data;
	}

	void UpdateHostFromDevice() override
	{
		this->data->UpdateHostFromDevice();
	}

	void UpdateDeviceFromHost() override
	{
		this->data->UpdateDeviceFromHost();
	}

	// Suppress the default copy constructor and assignment operator
	ITMTemplatedHierarchyLevel(const ITMTemplatedHierarchyLevel&);

	ITMTemplatedHierarchyLevel& operator=(const ITMTemplatedHierarchyLevel&);
};
}
