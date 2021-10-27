//
// Created by Malte Splietker on 05.02.21.
//

#pragma once

#include <ORUtils/Vector.h>

namespace ITMLib
{

typedef struct
{
	float sdfSum;
	float weightSum;
	Vector3f colorSum;
	float colorWeightSum;

	_CPU_AND_GPU_CODE_
	inline void reset()
	{
		sdfSum = 0.0f;
		weightSum = 0.0f;
		colorSum = Vector3f(0, 0, 0);
		colorWeightSum = 0.0f;
	}

	_CPU_AND_GPU_CODE_
	inline void update(float sdf, float weight, const Vector3f& color = Vector3f(0, 0, 0), float colorWeight = 0)
	{
#ifdef __CUDA_ARCH__
		atomicAdd(&sdfSum, weight * sdf);
		atomicAdd(&weightSum, weight);
		atomicAdd(&colorSum.x, colorWeight * color.x);
		atomicAdd(&colorSum.y, colorWeight * color.y);
		atomicAdd(&colorSum.z, colorWeight * color.z);
		atomicAdd(&colorWeightSum, colorWeight);
#else
		sdfSum += weight * sdf;
		weightSum += weight;
		colorSum += colorWeight * color;
		colorWeightSum += colorWeight;
#endif
	}
} SummingVoxel;

} // namespace ITMLib
