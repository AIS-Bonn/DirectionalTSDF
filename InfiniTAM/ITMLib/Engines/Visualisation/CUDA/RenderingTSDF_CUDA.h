//
// Created by Malte Splietker on 16.03.21.
//

#pragma once

#include <ITMLibDefines.h>
#include <stdgpu/unordered_map_fwd>
#include <ORUtils/Vector.h>

namespace ITMLib
{
typedef stdgpu::unordered_map<Vector3s, ITMVoxel*> RenderingTSDF;
}
