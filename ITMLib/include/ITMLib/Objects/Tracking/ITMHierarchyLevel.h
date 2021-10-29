//
// Created by Malte Splietker on 06.07.21.
//

#pragma once

#include "TrackerIterationType.h"
#include <ITMLib/Utils/ITMMath.h>

namespace ITMLib
{

/** Abstract class for representing hierarchy levels.
 *
 * A hierarchy level contains data of a certain level in an image hierarchy pyramid.
 */
class ITMHierarchyLevel
{
public:
	int levelId;

	TrackerIterationType iterationType;

	Vector4f intrinsics;

	ITMHierarchyLevel(int levelId, TrackerIterationType iterationType, bool skipAllocation = false)
		: levelId(levelId), iterationType(iterationType), manageData(!skipAllocation)
	{}

	virtual ~ITMHierarchyLevel() = default;

	ITMHierarchyLevel(const ITMHierarchyLevel&) = delete;

	ITMHierarchyLevel& operator=(const ITMHierarchyLevel&) = delete;

	virtual void UpdateHostFromDevice() = 0;

	virtual void UpdateDeviceFromHost() = 0;

protected:
	/** Flag to indicate, whether the data is allocated and thus freed by the object itself */
	bool manageData;
};

}