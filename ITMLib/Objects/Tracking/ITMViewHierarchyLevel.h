// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "TrackerIterationType.h"
#include "ITMHierarchyLevel.h"
#include "../../Utils/ITMMath.h"
#include "../../../ORUtils/Image.h"

namespace ITMLib
{
	class ITMViewHierarchyLevel : public ITMHierarchyLevel
	{
	public:
		ORUtils::Image<Vector4u> *rgb = nullptr;
		ORUtils::Image<float> *depth = nullptr;
		ORUtils::Image<Vector4s> *gradientX_rgb = nullptr, *gradientY_rgb = nullptr;

		ITMViewHierarchyLevel(Vector2i imgSize, int levelId, TrackerIterationType iterationType, MemoryDeviceType memoryType, bool skipAllocation)
		: ITMHierarchyLevel(levelId, iterationType, skipAllocation)
		{
			if (!skipAllocation) {
				this->rgb = new ORUtils::Image<Vector4u>(imgSize, memoryType);
				this->depth = new ORUtils::Image<float>(imgSize, memoryType);
				this->gradientX_rgb = new ORUtils::Image<Vector4s>(imgSize, memoryType);
				this->gradientY_rgb = new ORUtils::Image<Vector4s>(imgSize, memoryType);
			}
		}

		void UpdateHostFromDevice()
		{ 
			this->rgb->UpdateHostFromDevice();
			this->depth->UpdateHostFromDevice();
			this->gradientX_rgb->UpdateHostFromDevice();
			this->gradientY_rgb->UpdateHostFromDevice();
		}

		void UpdateDeviceFromHost()
		{ 
			this->rgb->UpdateDeviceFromHost();
			this->depth->UpdateDeviceFromHost();
			this->gradientX_rgb->UpdateDeviceFromHost();
			this->gradientY_rgb->UpdateDeviceFromHost();
		}

		~ITMViewHierarchyLevel(void)
		{
			if (manageData) {
				delete rgb;
				delete depth;
				delete gradientX_rgb; delete gradientY_rgb;
			}
		}

		// Suppress the default copy constructor and assignment operator
		ITMViewHierarchyLevel(const ITMViewHierarchyLevel&);
		ITMViewHierarchyLevel& operator=(const ITMViewHierarchyLevel&);
	};
}
