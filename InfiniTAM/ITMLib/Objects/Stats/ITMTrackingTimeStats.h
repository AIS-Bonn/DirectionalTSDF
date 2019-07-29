//
// Created by Malte Splietker on 26.07.19.
//

#pragma once

#include "ITMTimeStatsBase.h"

namespace ITMLib
{

struct ITMTrackingTimeStats : public ITMTimeStatsBase
{
	ITMTrackingTimeStats()
	{
		Reset();
	}

	void Reset() override
	{
		tracking = 0;
		rendering = 0;
	}

	float GetSum() const override
	{
		return tracking + rendering;
	}

	void Print(std::ostream &stream) const override
	{
		stream << tracking << " " << rendering;
	}

	void PrintHeader(std::ostream &stream) const override
	{
		stream << "tracking" << " " << "rendering";
	}

	float tracking;
	float rendering;
};

} // namespace ITMLib
