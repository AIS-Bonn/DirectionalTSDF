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
	FUSIONMODE_RAY_CASTING
} FusionMode;

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
	FusionMetric fusionMetric;
	bool useSpaceCarving;
	bool useWeighting;
};

} // namespace ITMLib

