//
// Created by Malte Splietker on 24.06.19.
//

#pragma once

#include "ITMLib/Utils/ITMStringUtils.h"

namespace ITMLib
{

typedef enum
{
	TSDFMODE_DEFAULT,
	TSDFMODE_DIRECTIONAL,
} TSDFMode;

typedef enum
{
	FUSIONMODE_VOXEL_PROJECTION,
	FUSIONMODE_RAY_CASTING_NORMAL,
	FUSIONMODE_RAY_CASTING_VIEW_DIR,
	FUSIONMODE_RAY_CASTING_VIEW_DIR_AND_NORMAL,
} FusionMode;

typedef enum
{
	CARVINGMODE_VOXEL_PROJECTION,
	CARVINGMODE_RAY_CASTING,
} CarvingMode;

typedef enum
{
	FUSIONMETRIC_POINT_TO_POINT,
	FUSIONMETRIC_POINT_TO_PLANE,
} FusionMetric;

inline TSDFMode TSDFModeFromString(const std::string &mode)
{
	if (iequals(mode, "default"))
		return TSDFMODE_DEFAULT;
	else if (iequals(mode, "directional"))
		return TSDFMODE_DIRECTIONAL;

	printf(R"(ERROR: Unknown TSDF mode "%s". Using "default" instead)", mode.c_str());
	return TSDFMODE_DEFAULT;
}

inline FusionMode FusionModeFromString(const std::string &mode)
{
	if (iequals(mode, "voxelProjection"))
		return FUSIONMODE_VOXEL_PROJECTION;
	else if (iequals(mode, "rayCastingNormal"))
		return FUSIONMODE_RAY_CASTING_NORMAL;
	else if (iequals(mode, "rayCastingViewDir"))
		return FUSIONMODE_RAY_CASTING_VIEW_DIR;
	else if (iequals(mode, "rayCastingViewDirAndNormal"))
		return FUSIONMODE_RAY_CASTING_VIEW_DIR_AND_NORMAL;

	printf(R"(ERROR: Unknown fusion mode "%s". Using "voxelProjection" instead)", mode.c_str());
	return FUSIONMODE_VOXEL_PROJECTION;
}

inline CarvingMode CarvingModeFromString(const std::string &mode)
{
	if (iequals(mode, "voxelProjection"))
		return CARVINGMODE_VOXEL_PROJECTION;
	else if (iequals(mode, "rayCasting"))
		return CARVINGMODE_RAY_CASTING;

	printf(R"(ERROR: Unknown carving mode "%s". Using "voxelProjection" instead)", mode.c_str());
	return CARVINGMODE_VOXEL_PROJECTION;
}

inline FusionMetric FusionMetricFromString(const std::string &metric)
{
	if (iequals(metric, "pointToPoint"))
		return FUSIONMETRIC_POINT_TO_POINT;
	else if (iequals(metric, "pointToPlane"))
		return FUSIONMETRIC_POINT_TO_PLANE;

	printf(R"(ERROR: Unknown fusion metric "%s". Using "pointToPoint" instead)", metric.c_str());
	return FUSIONMETRIC_POINT_TO_POINT;
}

class ITMFusionParams
{
public:
	TSDFMode tsdfMode;
	FusionMode fusionMode;
	CarvingMode carvingMode; // only relevant for ray casting fusion
	FusionMetric fusionMetric;
	bool useSpaceCarving;
	bool useWeighting;
};

} // namespace ITMLib

