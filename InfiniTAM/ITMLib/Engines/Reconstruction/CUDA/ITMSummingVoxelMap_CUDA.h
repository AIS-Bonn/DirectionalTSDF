//
// Created by Malte Splietker on 08.02.21.
//

#pragma once

#include <stdgpu/unordered_map.cuh>
#include <ITMLib/Engines/Reconstruction/Shared/ITMSummingVoxelMap.h>
#include <ITMLib/Utils/ITMTimer.h>


namespace ITMLib
{

__global__
void insertHashEntries_device(
	const ITMVoxelBlockHash::IndexData* hashTable,
	const int* visibleEntryIds,
	SummingVoxel* rayCastSum,
	stdgpu::unordered_map<BlockIndex, SummingVoxel*> summingVoxelMap
)
{
	int entryId = blockIdx.x;

	const ITMHashEntry& hashEntry = hashTable[visibleEntryIds[entryId]];
	if (not hashEntry.IsValid())
		return;

	BlockIndex idx(hashEntry.pos, hashEntry.direction == TSDFDirection_type(TSDFDirection::NONE) ? 0 : hashEntry.direction);
	auto it = summingVoxelMap.find(idx);
	if (it == summingVoxelMap.end())
	{
		summingVoxelMap.emplace(idx, rayCastSum + entryId * SDF_BLOCK_SIZE3);
	} else
	{
		it->second = rayCastSum + entryId * SDF_BLOCK_SIZE3;
	}
}

__global__
void clearSummingVoxels_device(SummingVoxel* rayCastSum)
{
	int locId = threadIdx.x;
	int entryId = blockIdx.x;

	rayCastSum[entryId * SDF_BLOCK_SIZE3 + locId].reset();
}

class SummingVoxelMap_CUDA : public SummingVoxelMap<stdgpu::unordered_map>
{
public:
	~SummingVoxelMap_CUDA()
	{
		ORcudaSafeCall(cudaFree(summingVoxels));
		if (map.bucket_count() > 0)
			stdgpu::unordered_map<BlockIndex, SummingVoxel*>::destroyDeviceObject(map);
	}

	inline void
	Init(const ITMVoxelBlockHash::IndexData* hashTable, const int* visibleEntryIds, int noVisibleEntries) override
	{
		if (noVisibleEntries <= 0)
			return;

		if (map.bucket_count() <= 0 or noVisibleEntries > allocationSize)
		{
			allocationSize = 2 * noVisibleEntries;
			Resize(allocationSize);
		}

		insertHashEntries_device << < noVisibleEntries, 1 >> > (
			hashTable, visibleEntryIds, summingVoxels, map);
		ORcudaKernelCheck;

		clearSummingVoxels_device << < noVisibleEntries, SDF_BLOCK_SIZE3 >> > (summingVoxels);
		ORcudaKernelCheck;
	}

private:
	size_t allocationSize;

	void Resize(int newSize)
	{
		ORcudaSafeCall(cudaFree(summingVoxels));
		if (map.bucket_count() > 0)
			stdgpu::unordered_map<BlockIndex, SummingVoxel*>::destroyDeviceObject(map);

		ORcudaSafeCall(cudaMalloc(&summingVoxels, newSize * SDF_BLOCK_SIZE3 * sizeof(SummingVoxel)));
		map = stdgpu::unordered_map<BlockIndex, SummingVoxel*>::createDeviceObject(
			newSize * 3.0); // factor 3 to prevent hash collisions
	}
};

} // namespace ITMLib
