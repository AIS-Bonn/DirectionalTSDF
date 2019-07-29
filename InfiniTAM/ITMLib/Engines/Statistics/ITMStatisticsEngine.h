//
// Created by Malte Splietker on 26.07.19.
//

#pragma once

#include <string>
#include <fstream>

#include "ITMLib/Objects/Stats/ITMTimeStats.h"

namespace ITMLib
{

class ITMStatisticsEngine
{
public:
	void Initialize(const std::string &outputDirectory);

	void CloseAll();

	void LogTimeStats(const ITMTimeStats &timeStats);

private:
	std::string m_outputDirectory;

	std::ofstream m_timeStatsFile;
};

} // namespace ITMLib
