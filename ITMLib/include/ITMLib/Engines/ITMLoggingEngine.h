//
// Created by Malte Splietker on 26.07.19.
//

#pragma once

#include <string>
#include <fstream>

#include <ITMLib/Objects/Stats/ITMTimeStats.h>
#include <ITMLib/Objects/Tracking/ITMTrackingState.h>

namespace ITMLib
{

class ITMLoggingEngine
{
public:
	void Initialize(const std::string& outputDirectory);

	void CloseAll();

	void LogTimeStats(const ITMTimeStats& timeStats);

	void LogPose(const ITMTrackingState& trackingState);

	void LogBlockAllocations(const unsigned int* noAllocationsPerDirection);

private:
	std::string m_outputDirectory;

	std::ofstream m_timeStatsFile;

	std::ofstream m_trackingFile;

	std::ofstream m_allocationsFile;
};

} // namespace ITMLib
