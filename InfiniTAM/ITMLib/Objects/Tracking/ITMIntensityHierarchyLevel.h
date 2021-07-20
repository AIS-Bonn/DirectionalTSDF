// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMHierarchyLevel.h"
#include "TrackerIterationType.h"
#include "../../Utils/ITMMath.h"
#include "../../../ORUtils/Image.h"

namespace ITMLib
{
	class ITMIntensityHierarchyLevel : public ITMHierarchyLevel
	{
	public:
		ORUtils::Image<float> *intensity_current = nullptr;
		ORUtils::Image<float> *intensity_prev = nullptr;

		/** Gradient of intensity_prev used for computing the Jacobian */
		ORUtils::Image<Vector2f> *gradients = nullptr;

		ITMIntensityHierarchyLevel(Vector2i imgSize, int levelId, TrackerIterationType iterationType,
			MemoryDeviceType memoryType, bool skipAllocation = false)
			: ITMHierarchyLevel(levelId, iterationType, skipAllocation)
		{
			if (!skipAllocation)
			{
				this->intensity_current = new ORUtils::Image<float>(imgSize, memoryType);
				this->intensity_prev = new ORUtils::Image<float>(imgSize, memoryType);
			}

			this->gradients = new ORUtils::Image<Vector2f>(imgSize, memoryType);
		}

		void UpdateHostFromDevice() override
		{ 
			if (!this->intensity_current || !this->intensity_prev)
				throw std::runtime_error("ITMIntensityHierarchyLevel: did not set intensity images.");

			this->intensity_current->UpdateHostFromDevice();
			this->intensity_prev->UpdateHostFromDevice();
			this->gradients->UpdateHostFromDevice();
		}

		void UpdateDeviceFromHost() override
		{ 
			if (!this->intensity_current || !this->intensity_prev)
				throw std::runtime_error("ITMIntensityHierarchyLevel: did not set intensity images.");

			this->intensity_current->UpdateDeviceFromHost();
			this->intensity_prev->UpdateDeviceFromHost();
			this->gradients->UpdateDeviceFromHost();
		}

		~ITMIntensityHierarchyLevel(void)
		{
			if (manageData)
			{
				delete intensity_current;
				delete intensity_prev;
			}

			delete gradients;
		}

		// Suppress the default copy constructor and assignment operator
		ITMIntensityHierarchyLevel(const ITMIntensityHierarchyLevel&);
		ITMIntensityHierarchyLevel& operator=(const ITMIntensityHierarchyLevel&);
	};
}
