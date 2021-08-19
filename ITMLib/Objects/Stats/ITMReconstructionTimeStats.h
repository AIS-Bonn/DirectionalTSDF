//
// Created by Malte Splietker on 26.07.19.
//

#pragma once

#include "ITMTimeStatsBase.h"

namespace ITMLib
{

struct ITMReconstructionTimeStats : public ITMTimeStatsBase
{
	ITMReconstructionTimeStats()
	{
		Reset();
	}

	void Reset() override
	{
		buildingVisibilityList = 0;
		allocation = 0;
		fusion = 0;
		carving = 0;
		swapping = 0;
	}

	float GetSum() const override
	{
		return buildingVisibilityList + allocation + fusion + carving + swapping;
	}

	void Print(std::ostream& stream) const override
	{
		stream << buildingVisibilityList << " " << allocation << " " << fusion
		       << " " << carving << " " << swapping;
	}

	void PrintHeader(std::ostream &stream) const override
	{
		stream << "buildingVisibilityList" << " " << "allocation" << " " << "fusion"
		       << " " << "carving" << " " << "swapping";
	}

	float buildingVisibilityList;
	float allocation;
	float fusion;
	float carving;
	float swapping;
};

} // namespace ITMLib
