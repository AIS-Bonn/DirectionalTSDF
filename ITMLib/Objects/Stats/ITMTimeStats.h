//
// Created by Malte Splietker on 26.07.19.
//

#pragma once

#include <ostream>

#include "ITMReconstructionTimeStats.h"
#include "ITMPreprocessingTimeStats.h"
#include "ITMRelocalizationTimeStats.h"
#include "ITMTrackingTimeStats.h"

namespace ITMLib
{

struct ITMTimeStats : public ITMTimeStatsBase
{
	ITMPreprocessingTimeStats preprocessing;

	ITMTrackingTimeStats tracking;

	ITMRelocalizationTimeStats relocalization;

	ITMReconstructionTimeStats reconstruction;

	void Reset() override
	{
		preprocessing.Reset();
		tracking.Reset();
		relocalization.Reset();
		reconstruction.Reset();
	}

	float GetSum() const override
	{
		return preprocessing.GetSum() + tracking.GetSum() + relocalization.GetSum() + reconstruction.GetSum();
	}

	void Print(std::ostream& stream) const override
	{
		stream << preprocessing << " " << tracking << " " << relocalization << " " << reconstruction;
	}

	void PrintHeader(std::ostream& stream) const override
	{
		stream << "# ";
		preprocessing.PrintHeader(stream);
		stream << " ";
		tracking.PrintHeader(stream);
		stream << " ";
		relocalization.PrintHeader(stream);
		stream << " ";
		reconstruction.PrintHeader(stream);
		stream << std::endl;
	}
};

} // namespace ITMLib
