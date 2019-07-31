//
// Created by Malte Splietker on 26.07.19.
//

#include <iomanip>
#include <experimental/filesystem>

#include "ITMLoggingEngine.h"

namespace fs = std::experimental::filesystem;

namespace ITMLib
{

void ITMLoggingEngine::Initialize(const std::string& outputDirectory)
{
	m_outputDirectory = outputDirectory;
	fs::create_directories(m_outputDirectory);

	m_timeStatsFile.open(m_outputDirectory + "/time_stats.txt");
	m_timeStatsFile.flags(std::ios::right);
	m_timeStatsFile.setf(std::ios::fixed);
	ITMTimeStats stats_dummy;
	stats_dummy.PrintHeader(m_timeStatsFile);

	m_trackingFile.open(m_outputDirectory + "/tracking.txt");
	m_trackingFile << std::fixed << std::left << std::setprecision(10);
	m_trackingFile << "# tx ty tz qx qy qz qw score result" << std::endl;
}

void ITMLoggingEngine::CloseAll()
{
	m_timeStatsFile.flush();
	m_timeStatsFile.close();

	m_trackingFile.flush();
	m_trackingFile.close();
}

void ITMLoggingEngine::LogTimeStats(const ITMTimeStats& timeStats)
{
	m_timeStatsFile << timeStats << std::endl;
}

void ITMLoggingEngine::LogPose(const ITMTrackingState& trackingState)
{
	Matrix4f invM = trackingState.pose_d->GetInvM();
	Vector3f T;
	for (int i = 0; i < 3; ++i) T[i] = invM.m[3 * 4 + i];

	Vector4f Q = trackingState.pose_d->GetQ();

	m_trackingFile << T.x << " " << T.y << " " << T.z << " " << Q.x << " " << Q.y << " " << Q.z << " " << Q.w;
	m_trackingFile << " " << trackingState.trackerScore;
	m_trackingFile << " " << trackingState.trackerResult;
	m_trackingFile << std::endl;
}

} // namespace ITMLib