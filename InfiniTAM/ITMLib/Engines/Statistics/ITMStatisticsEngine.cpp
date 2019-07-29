//
// Created by Malte Splietker on 26.07.19.
//

#include <experimental/filesystem>

#include "ITMStatisticsEngine.h"

namespace fs = std::experimental::filesystem;

namespace ITMLib
{

void ITMStatisticsEngine::Initialize(const std::string& outputDirectory)
{
	m_outputDirectory = outputDirectory;
	fs::create_directories(m_outputDirectory);

	m_timeStatsFile.open(m_outputDirectory + "/time_stats.txt");
	m_timeStatsFile.flags(std::ios::right);
	m_timeStatsFile.setf(std::ios::fixed);
	ITMTimeStats stats_dummy;
	stats_dummy.PrintHeader(m_timeStatsFile);
}

void ITMStatisticsEngine::CloseAll()
{
	m_timeStatsFile.flush();
	m_timeStatsFile.close();
}

void ITMStatisticsEngine::LogTimeStats(const ITMTimeStats& timeStats)
{
	m_timeStatsFile << timeStats << std::endl;
}

} // namespace ITMLib