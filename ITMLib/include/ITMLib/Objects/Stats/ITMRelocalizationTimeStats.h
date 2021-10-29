//
// Created by Malte Splietker on 26.07.19.
//

#pragma once

#include "ITMTimeStatsBase.h"

namespace ITMLib
{

struct ITMRelocalizationTimeStats : public ITMTimeStatsBase
{
	ITMRelocalizationTimeStats()
	{
		Reset();
	}

	void Reset() override
	{
		relocalization = 0;
	}

	float GetSum() const override
	{
		return 0;
	}

	void Print(std::ostream& stream) const override
	{
		stream << relocalization;
	}

	void PrintHeader(std::ostream& stream) const override
	{
		stream << "relocalization";
	}

	float relocalization;
};

} // namespace ITMLib
