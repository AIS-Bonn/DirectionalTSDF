//
// Created by Malte Splietker on 26.07.19.
//

#pragma once

#include "ITMTimeStatsBase.h"

namespace ITMLib
{

struct ITMPreprocessingTimeStats : public ITMTimeStatsBase
{
	ITMPreprocessingTimeStats()
	{
		Reset();
	}

	void Reset() override
	{
		copyImages = 0;
		bilateralFilter = 0;
		normalEstimation = 0;
	}

	float GetSum() const override
	{
		return copyImages + bilateralFilter + normalEstimation;
	}

	void Print(std::ostream &stream) const override
	{
		stream << copyImages << " " << bilateralFilter << " " << normalEstimation;
	}

	void PrintHeader(std::ostream &stream) const override
	{
		stream << "copyImages" << " " << "bilateralFilter" << " " << "normalEstimation";
	}

	float copyImages;
	float bilateralFilter;
	float normalEstimation;
};

} // namespace ITMLib
