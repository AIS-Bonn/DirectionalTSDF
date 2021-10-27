//
// Created by Malte Splietker on 26.07.19.
//

#include <iomanip>
#include <experimental/filesystem>
#include <iostream>
#include <ITMLib/Objects/Scene/ITMDirectional.h>

#include "ITMLoggingEngine.h"

namespace fs = std::experimental::filesystem;

namespace ITMLib
{

void ITMLoggingEngine::Initialize(const std::string& outputDirectory)
{
	m_outputDirectory = outputDirectory;
	fs::create_directories(m_outputDirectory);

	m_timeStatsFile.open(m_outputDirectory + "/time_stats.txt");
	if (not m_timeStatsFile.is_open())
		std::cerr << "failed to open " << m_outputDirectory + "/time_stats.txt" << " for writing" << std::endl;
	m_timeStatsFile.flags(std::ios::right);
	m_timeStatsFile.setf(std::ios::fixed);
	ITMTimeStats stats_dummy;
	stats_dummy.PrintHeader(m_timeStatsFile);

	m_trackingFile.open(m_outputDirectory + "/tracking.txt");
	if (not m_trackingFile.is_open())
		std::cerr << "failed to open " << m_outputDirectory + "/tracking.txt" << " for writing" << std::endl;
	m_trackingFile << std::fixed << std::left << std::setprecision(10);
	m_trackingFile << "# tx ty tz qx qy qz qw score result" << std::endl;

	m_allocationsFile.open(m_outputDirectory + "/allocation.txt");
	if (not m_allocationsFile.is_open())
		std::cerr << "failed to open " << m_outputDirectory + "/allocation.txt" << " for writing" << std::endl;
	m_allocationsFile << "# X_POS X_NEG Y_POS Y_NEG Z_POS Z_NEG" << std::endl;
}

void ITMLoggingEngine::CloseAll()
{
	m_timeStatsFile.flush();
	m_timeStatsFile.close();

	m_trackingFile.flush();
	m_trackingFile.close();

	m_allocationsFile.flush();
	m_allocationsFile.close();
}

void ITMLoggingEngine::LogTimeStats(const ITMTimeStats& timeStats)
{
	m_timeStatsFile << timeStats << std::endl;
}

Vector4f QuaternionFromTransformationMatrix(const Matrix4f& M)
{
	Matrix3f R;
	R.m00 = M.m00; R.m10 = M.m10; R.m20 = M.m20;
	R.m01 = M.m01; R.m11 = M.m11; R.m21 = M.m21;
	R.m02 = M.m02; R.m12 = M.m12; R.m22 = M.m22;
	float values[4]; // x, y, z, w

	// "Quaternion Calculus and Fast Animation", Ken Shoemake, 1987 SIGGRAPH course notes
	float trace = R.m00 + R.m11 + R.m22;
	if (trace > 0)
	{
		trace = sqrt(1 + trace);
		float s = 1 / (2 * trace);
		values[3] = 0.5 * trace;
		values[0] = (R.m12 - R.m21) * s;
		values[1] = (R.m20 - R.m02) * s;
		values[2] = (R.m01 - R.m10) * s;
	}
	else
	{
		int i = 0;
		if (R.m11 > R.m00)
			i = 1;
		if (R.m22 > R.at(i, i))
			i = 2;
		int j = (i+1)%3;
		int k = (j+1)%3;

		trace = sqrt(R.at(i,i) - R.at(j,j) - R.at(k,k) + 1.0);
		float s = 1 / (2 * trace);
		values[i] = 0.5 * trace;
		values[3] = (R.at(j, k) - R.at(k, j)) * s;
		values[j] = (R.at(i, j) + R.at(j, i)) * s;
		values[k] = (R.at(i, k) + R.at(k, i)) * s;
	}
	return Vector4f(values[0], values[1], values[2], values[3]);
}

void ITMLoggingEngine::LogPose(const ITMTrackingState& trackingState)
{
	Matrix4f T_WC = trackingState.pose_d->GetInvM();
	Vector3f T = (T_WC * Vector4f(0, 0, 0, 1)).toVector3();
	Vector4f Q = QuaternionFromTransformationMatrix(T_WC);

	m_trackingFile << T.x << " " << T.y << " " << T.z << " " << Q.x << " " << Q.y << " " << Q.z << " " << Q.w;
	m_trackingFile << " " << trackingState.trackerScore;
	m_trackingFile << " " << trackingState.trackerResult;
	m_trackingFile << std::endl;
}

void ITMLoggingEngine::LogBlockAllocations(const unsigned int* noAllocationsPerDirection)
{
	if (not noAllocationsPerDirection)
		return;

	m_allocationsFile << noAllocationsPerDirection[0];
	for (size_t d = 1; d < N_DIRECTIONS; d++)
		m_allocationsFile << " " << noAllocationsPerDirection[d];
	m_allocationsFile << std::endl;
}

} // namespace ITMLib