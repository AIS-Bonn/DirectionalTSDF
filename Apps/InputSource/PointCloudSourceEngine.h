//
// Created by Malte Splietker on 18.10.21.
//

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <ITMLib/Utils/ITMMath.h>
#include <ORUtils/Image.h>
#include <ITMLib/Objects/Misc/ITMPointCloud.h>

#pragma once

namespace InputSource
{

class PointCloudSourceEngine
{
public:
	PointCloudSourceEngine(const std::string& directory);

	void ReadPLY(const std::string& filepath);

	void ReadKITTI(const std::string& path)
	{
		std::string pcFile;
		std::ifstream stream(pcFile, std::ios::binary);
		if (!stream)
		{
			std::cerr << "failed to open " << pcFile << std::endl;
			return;
		}

		uint32_t numPoints;
		stream.read(reinterpret_cast<char*>(&numPoints), sizeof(int32_t));
		numPoints /= 4;


		ORUtils::Image<ITMLib::Vector4f> points(ITMLib::Vector2i(1, 1), MEMORYDEVICE_CPU);
		float* ptr = reinterpret_cast<float*>(points.GetData(MEMORYDEVICE_CPU));

		for (size_t i = 0; i < numPoints; i++)
		{
			stream.read(reinterpret_cast<char*>(&ptr[i * 4]), 4 * sizeof(float));
		}
		stream.close();

//		currentPointCloud->locations->SetFrom(points);
	}

//	virtual ITMLib::ITMRGBDCalib getCalib() const = 0;
//
//	virtual void getPointCloud(ITMLib::Vector4f* currentPointCloud, ITMLib::Vector4f* colors) = 0;
//
//	/**
//	 * \brief Gets the size of the next depth image (if any).
//	 *
//	 * \pre     hasMoreImages()
//	 * \return  The size of the next depth image (if any), or Vector2i(0,0) otherwise.
//	 */
//	virtual Vector2i getDepthImageSize(void) const = 0;

	void getPointCloud(ITMLib::ITMPointCloud* pointCloud);

	inline bool hasMoreData() const
	{
		return currentPointCloudIdx < pointCloudPaths.size();
	};

private:
	ITMLib::ITMPointCloud* currentPointCloud = nullptr;
	std::vector<std::filesystem::path> pointCloudPaths;
	size_t currentPointCloudIdx = 0;
};

};