//
// Created by Malte Splietker on 09.04.21.
//

#pragma once

#include <unordered_map>
#include <stdgpu/memory.h>

namespace std
{

template<typename T>
struct hash<ITMLib::IndexDirectional<T>>
{
	inline std::size_t operator()(const ITMLib::IndexDirectional<T>& k) const
	{
		return ((std::hash<T>()(k.x)
		         ^ (std::hash<T>()(k.y) << 1)) >> 1)
		       ^ (std::hash<T>()(k.z) << 1)
		       ^ (std::hash<T>()(k.w) << 1);
	}
};

template<typename T>
struct hash<ITMLib::Index<T>>
{
	inline std::size_t operator()(const ITMLib::Index<T>& k) const
	{
		return ((std::hash<T>()(k.x)
		         ^ (std::hash<T>()(k.y) << 1)) >> 1)
		       ^ (std::hash<T>()(k.z) << 1);
	}
};

}

namespace ITMLib
{

template<typename TIndex, typename TVoxel>
class TSDF_CPU : public TSDFBase<TIndex, TVoxel, std::unordered_map>
{
public:
	~TSDF_CPU()
	{
		delete this->voxels;
	}

	void allocate(const TIndex* blocks, size_t N) override
	{
		if (N <= 0)
			return;

		if (this->size() >= this->allocatedBlocksMax)
		{
			printf("warning: TSDF size exceeded (%zu/%zu allocated). stopped allocating.\n", this->size(),
			       this->allocatedBlocksMax);
			return;
		}

		for (size_t i = 0; i < N; i++)
		{
			const TIndex& block = blocks[i];
			if (this->map.find(block) != this->map.end())
				continue;

			if (block.getDirection() != TSDFDirection::NONE)
			{
				this->allocationStats.noAllocationsPerDirection[static_cast<TSDFDirection_type>(block.getDirection())]++;
			} else
			{
				this->allocationStats.noAllocationsPerDirection[0]++;
			}

			size_t offset = this->allocationStats.noAllocations;
			this->allocationStats.noAllocations++;
			this->map.emplace(block, this->voxels + offset * SDF_BLOCK_SIZE3);
		}
	}

	void resize(size_t newSize) override
	{
		if (newSize <= 0)
			return;
		this->allocatedBlocksMax = newSize;

		delete this->voxels;
		this->voxels = (TVoxel*) malloc(newSize * SDF_BLOCK_SIZE3 * sizeof(TVoxel));
		std::fill_n(this->voxels, newSize * SDF_BLOCK_SIZE3, TVoxel());

		this->map.reserve(newSize);
		this->map.clear();
		this->allocationStats = AllocationStats();
	}

	MemoryDeviceType deviceType() override
	{ return MemoryDeviceType::MEMORYDEVICE_CUDA; }

	explicit TSDF_CPU(size_t size = 1)
	{
		resize(size);
	}
};


template<typename TIndex, typename TVoxel>
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
