//
// Created by Malte Splietker on 24.06.19.
//

#pragma once

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
	FUSIONMODE_RAY_CASTING_VIEW_DIR_AND_NORMAL
} FusionMode;

typedef enum
{
	CARVINGMODE_VOXEL_PROJECTION,
	CARVINGMODE_RAY_CASTING
} CarvingMode;

typedef enum
{
	FUSIONMETRIC_POINT_TO_POINT,
	FUSIONMETRIC_POINT_TO_PLANE
} FusionMetric;

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

