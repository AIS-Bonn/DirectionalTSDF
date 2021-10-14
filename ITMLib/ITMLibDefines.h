// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <ITMLib/Objects/Scene/ITMScene.h>
#include "Objects/Scene/ITMVoxelBlockHash.h"
#include "Objects/Scene/ITMVoxelTypes.h"

/** This chooses the information stored at each voxel. At the moment, valid
    options are ITMVoxel_s, ITMVoxel_f, ITMVoxel_s_rgb and ITMVoxel_f_rgb.
*/
typedef ITMVoxel_f_rgb ITMVoxel;

/** This chooses the way the voxels are addressed and indexed. At the moment,
    valid options are ITMVoxelBlockHash and ITMPlainVoxelArray.
*/
typedef ITMLib::ITMVoxelBlockHash ITMVoxelIndex;

/**
 * Index used for TSDF data structure. To keep the code simple, directional is used even for non-directional mode
 */
typedef ITMLib::IndexDirectionalShort ITMIndex;
typedef ITMLib::IndexShort ITMIndexXYZ;

typedef ITMLib::ITMScene<ITMVoxel> Scene;