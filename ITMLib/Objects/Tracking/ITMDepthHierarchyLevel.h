// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMHierarchyLevel.h"
#include "TrackerIterationType.h"
#include "../../Utils/ITMMath.h"
#include "../../../ORUtils/Image.h"

namespace ITMLib
{
class ITMDepthHierarchyLevel : public ITMHierarchyLevel
{
public:
	ORUtils::Image<float>* depth = nullptr;

	ITMDepthHierarchyLevel(Vector2i imgSize, int levelId, TrackerIterationType iterationType,
	                       MemoryDeviceType memoryType, bool skipAllocation = false)
		: ITMHierarchyLevel(levelId, iterationType, skipAllocation)
	{
		if (!skipAllocation)
		{
			this->depth = new ORUtils::Image<float>(imgSize, memoryType);
		}
	}

	void UpdateHostFromDevice() override
	{
		this->depth->UpdateHostFromDevice();
	}

	void UpdateDeviceFromHost() override
	{
		this->depth->UpdateDeviceFromHost();
	}

	~ITMDepthHierarchyLevel(void)
	{
		if (manageData)
			delete depth;
	}

	// Suppress the default copy constructor and assignment operator
	ITMDepthHierarchyLevel(const ITMDepthHierarchyLevel&);

	ITMDepthHierarchyLevel& operator=(const ITMDepthHierarchyLevel&);
};
}
