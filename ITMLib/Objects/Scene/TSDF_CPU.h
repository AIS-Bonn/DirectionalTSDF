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

template <typename TIndex, typename TVoxel>
class TSDF_CPU : public TSDFBase<TIndex, TVoxel, std::unordered_map>
{
public:
	~TSDF_CPU()
	{
//		ORcudaSafeCall(cudaFree(this->voxels));
//		if (this->map.bucket_count() > 0)
//			stdgpu::unordered_map<TIndex, TVoxel*>::destroyDeviceObject(this->map);
	}

	void clear()
	{

	}

	size_t size()
	{
		return this->map.size();
	}

	virtual void allocate(const TIndex* blocks, size_t N)
	{
		printf("error: not implemented\n");
	}

	void resize(size_t newSize) override
	{
		if (newSize <= 0)
			return;
		this->allocatedBlocksMax = newSize;

		if (this->voxels)
			destroyHostArray(this->voxels);
		this->voxels = createHostArray(newSize * SDF_BLOCK_SIZE3, TVoxel());
		this->map.clear();
	}

	explicit TSDF_CPU(size_t size)
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
