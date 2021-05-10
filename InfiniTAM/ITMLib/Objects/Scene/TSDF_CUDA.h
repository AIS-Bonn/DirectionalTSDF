//
// Created by Malte Splietker on 08.04.21.
//

#pragma once

#include <ORUtils/CUDADefines.h>
#include <ITMLib/Core/ITMConstants.h>
#include <ITMLib/Objects/Scene/TSDF.h>
#include <ITMLib/Utils/ITMCUDAUtils.h>
#include <ITMLib/Utils/ITMGeometry.h>
#include <stdgpu/unordered_map.cuh>

namespace stdgpu{

template <typename T>
struct hash<ITMLib::IndexDirectional<T>>
{
	inline STDGPU_HOST_DEVICE
	stdgpu::index_t operator()(const ITMLib::IndexDirectional<T>& k) const
	{
		return ITMLib::hash(11536487, k.x)
		       + ITMLib::hash(14606887, k.y)
		       + ITMLib::hash(28491781, k.z)
		       + ITMLib::hash(83492791, static_cast<uint8_t>(k.w));
	}
};

template <typename T>
struct hash<ITMLib::Index<T>>
{
	inline STDGPU_HOST_DEVICE
	stdgpu::index_t operator()(const ITMLib::Index<T>& k) const
	{
		return ITMLib::hash(11536487, k.x)
		       + ITMLib::hash(14606887, k.y)
		       + ITMLib::hash(28491781, k.z);
	}
};

}

namespace ITMLib
{

template<typename TVoxel>
__global__
void clearVoxels_device(TVoxel* voxels)
{
	int locId = threadIdx.x;
	int entryId = blockIdx.x;

	voxels[entryId * SDF_BLOCK_SIZE3 + locId] = TVoxel();
}

template<typename TIndex, typename TVoxel, template<typename, typename...> class Map, typename... Args>
__global__ void
allocateBlocks_device(Map<TIndex, TVoxel*, Args...> tsdf, AllocationStats *allocationStats, TVoxel* voxels, const TIndex* allocationBlocksList, const size_t N)
{
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= N)
		return;

	if (tsdf.find(allocationBlocksList[index]) != tsdf.end())
		return; // already allocated

	if (allocationBlocksList[index].getDirection() != TSDFDirection::NONE)
	{
		atomicAdd(allocationStats->noAllocationsPerDirection +
		          static_cast<TSDFDirection_type>(allocationBlocksList[index].getDirection()), 1);
	}
	else
	{
		atomicAdd(allocationStats->noAllocationsPerDirection, 1);
	}

	size_t offset = atomicAdd(&(allocationStats->noAllocations), 1);
	tsdf.emplace(allocationBlocksList[index], voxels + offset * SDF_BLOCK_SIZE3);
}

template <typename TIndex, typename TVoxel>
class TSDF_CUDA : public TSDFBase<TIndex, TVoxel, stdgpu::unordered_map>
{
public:
	~TSDF_CUDA()
	{
//		ORcudaSafeCall(cudaFree(this->voxels));
//		if (this->map.bucket_count() > 0)
//			stdgpu::unordered_map<TIndex, TVoxel*>::destroyDeviceObject(this->map);
	}

	inline void resize(size_t newSize) override
	{
		if (newSize <= 0)
			return;
		this->allocatedBlocksMax = newSize;

		if (this->voxels)
			destroyDeviceArray(this->voxels);
		if (this->map.bucket_count() > 0)
			stdgpu::unordered_map<TIndex, TVoxel*>::destroyDeviceObject(this->map);

		this->voxels = createDeviceArray(newSize * SDF_BLOCK_SIZE3, TVoxel());
		this->map = stdgpu::unordered_map<TIndex, TVoxel*>::createDeviceObject(
			newSize * 3.0); // factor 3 to prevent hash bucket overflow

		clearVoxels_device << < this->allocatedBlocksMax, SDF_BLOCK_SIZE3 >> > (this->voxels);
		ORcudaKernelCheck;

		this->allocationStats = AllocationStats();
	}

	inline void allocate(const TIndex* blocks, size_t N) override
	{
		if (N > 0 and this->size() > this->allocatedBlocksMax)
		{
			printf("warning: TSDF size exceeded (%i/%zu allocated). stopped allocating.\n", this->size(), this->allocatedBlocksMax);
			return;
		}
		if (N <= 0)
			return;

		ORcudaSafeCall(cudaMemcpy(allocationStats_device, &this->allocationStats, sizeof(AllocationStats), cudaMemcpyHostToDevice));

		dim3 blockSize(256, 1);
		dim3 gridSize((int) ceil((float) N / (float) blockSize.x));
		allocateBlocks_device<<<blockSize, gridSize>>>(this->getMap(), allocationStats_device, this->voxels, blocks, N);
		ORcudaKernelCheck;

		ORcudaSafeCall(cudaMemcpy(&this->allocationStats, allocationStats_device, sizeof(AllocationStats), cudaMemcpyDeviceToHost));
	}

	explicit TSDF_CUDA(size_t size)
	{
		resize(size);
		ORcudaSafeCall(cudaMalloc(&allocationStats_device, sizeof(AllocationStats)));
		ORcudaSafeCall(cudaMemcpy(allocationStats_device, &this->allocationStats, sizeof(AllocationStats), cudaMemcpyHostToDevice));
	}

private:
	AllocationStats* allocationStats_device;
};

template<typename TIndex, typename TVoxel>
_CPU_AND_GPU_CODE_
inline TVoxel readVoxel(bool& found, const stdgpu::unordered_map<TIndex, TVoxel*>& tsdf,
                        const Vector3i& voxelIdx, const TSDFDirection direction = TSDFDirection::NONE)
{
#ifndef __CUDA_ARCH__
	fprintf(stderr, "error: no CPU access on TSDF_CUDA\n");
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