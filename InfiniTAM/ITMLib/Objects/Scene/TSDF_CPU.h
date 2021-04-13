//
// Created by Malte Splietker on 09.04.21.
//

#pragma once

#include <unordered_map>
#include <stdgpu/memory.h>

namespace ITMLib
{

template <typename TVoxel, typename TIndex>
class TSDF_CPU : public TSDF<TVoxel, TIndex, std::unordered_map>
{
public:
	~TSDF_CPU()
	{
//		ORcudaSafeCall(cudaFree(this->voxels));
//		if (this->map.bucket_count() > 0)
//			stdgpu::unordered_map<TIndex, TVoxel*>::destroyDeviceObject(this->map);
	}

	void Resize(size_t newSize) override
	{
		if (newSize <= 0)
			return;
		this->allocationSize = newSize;

		if (this->voxels)
			destroyHostArray(this->voxels);
		this->voxels = createHostArray(newSize * SDF_BLOCK_SIZE3, TVoxel());
		this->map.clear();
	}

	explicit TSDF_CPU(size_t size)
	{
		Resize(size);
	}
};


template<typename TVoxel, typename TIndex>
_CPU_AND_GPU_CODE_
inline TVoxel readVoxel(bool& found, const std::unordered_map<TIndex, TVoxel*>& tsdf,
                        const Vector3i& voxelIdx, const TSDFDirection direction = TSDFDirection::NONE)
{
#ifdef __CUDA_ARCH__
	printf("error: no CUDA access on TSDF_CPU\n");
	return TVoxel();
#else
	TIndex index;
		unsigned short linearIdx;
		voxelIdxToIndexAndOffset(index, linearIdx, voxelIdx, direction);

		auto it = tsdf.find(index);
		if (it == tsdf.end())
		{
			found = false;
			return TVoxel();
		}

		found = true;
		return it->second[linearIdx];
#endif
}

} // namespace ITMlib
