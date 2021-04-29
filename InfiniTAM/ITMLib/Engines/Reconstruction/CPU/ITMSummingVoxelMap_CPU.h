//
// Created by Malte Splietker on 10.02.21.
//

#pragma once

#include <unordered_map>
#include <ITMLib/Engines/Reconstruction/Shared/ITMSummingVoxelMap.h>

namespace ITMLib
{

class SummingVoxelMap_CPU : public SummingVoxelMap<std::unordered_map>
{
public:
	~SummingVoxelMap_CPU()
	{
		free(summingVoxels);
	}

	inline void Init(const ITMVoxelBlockHash::IndexData* hashTable, const int* visibleEntryIds, int noVisibleEntries) override
	{
		if (noVisibleEntries <= 0)
			return;

		if(not summingVoxels or noVisibleEntries > allocationSize)
		{
			allocationSize = 2 * noVisibleEntries;
			Resize(allocationSize);
		}

		insertAndResetHashEntries(hashTable, visibleEntryIds, noVisibleEntries);
	}

private:
	size_t allocationSize;

	void Resize(int newSize)
	{
		free(summingVoxels);
		summingVoxels = static_cast<SummingVoxel*>(malloc(newSize * SDF_BLOCK_SIZE3 * sizeof(SummingVoxel)));
	}

	void insertAndResetHashEntries(
		const ITMVoxelBlockHash::IndexData* hashTable,
		const int* visibleEntryIds,
		const int noVisibleEntries
	)
	{
		for (int entryId = 0; entryId < noVisibleEntries; entryId++)
		{
			const ITMHashEntry& hashEntry = hashTable[visibleEntryIds[entryId]];
			if (not hashEntry.IsValid())
				return;

			IndexType idx(hashEntry.pos, hashEntry.direction);
			auto it = map.find(idx);
			if (it == map.end())
			{
				auto ptr = map.emplace(idx, summingVoxels + entryId * SDF_BLOCK_SIZE3);
				ptr.first->second->reset();
			}
			else
			{
				it->second = summingVoxels + entryId * SDF_BLOCK_SIZE3;
				it->second->reset();
			}
		}
	}
};

} // namespace ITMLib
