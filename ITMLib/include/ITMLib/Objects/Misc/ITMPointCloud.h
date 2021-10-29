// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../Utils/ITMMath.h"
#include <ORUtils/Image.h>

namespace ITMLib
{
class ITMPointCloud
{
public:
	uint noTotalPoints;

	ORUtils::Image<Vector4f>* locations, * normals, * colours;

	explicit ITMPointCloud(Vector2i imgSize, MemoryDeviceType memoryType)
	{
		this->noTotalPoints = 0;

		locations = new ORUtils::Image<Vector4f>(imgSize, memoryType);
		colours = new ORUtils::Image<Vector4f>(imgSize, memoryType);
		normals = new ORUtils::Image<Vector4f>(imgSize, memoryType);
	}

	void Resize(const Vector2i size)
	{
		locations->ChangeDims(size);
		colours->ChangeDims(size);
		normals->ChangeDims(size);
	}

	void UpdateHostFromDevice()
	{
		locations->UpdateHostFromDevice();
		colours->UpdateHostFromDevice();
		normals->UpdateHostFromDevice();
	}

	void UpdateDeviceFromHost()
	{
		locations->UpdateDeviceFromHost();
		colours->UpdateDeviceFromHost();
		normals->UpdateDeviceFromHost();
	}

	~ITMPointCloud()
	{
		delete locations;
		delete colours;
		delete normals;
	}

	// Suppress the default copy constructor and assignment operator
	ITMPointCloud(const ITMPointCloud&);

	ITMPointCloud& operator=(const ITMPointCloud&);
};
}
