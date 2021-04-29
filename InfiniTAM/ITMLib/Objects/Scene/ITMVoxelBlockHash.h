// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <stdlib.h>
#include <fstream>
#include <iostream>

#include "ITMLib/Core/ITMConstants.h"
#include "ITMLib/Utils/ITMMath.h"
#include "ORUtils/MemoryBlock.h"
#include "ORUtils/MemoryBlockPersister.h"
#include "ITMLib/Objects/Scene/ITMDirectional.h"

/** \brief
	A single entry in the hash table.
*/
struct ITMHashEntry
{
	/** Position of the corner of the 8x8x8 volume, that identifies the entry. */
	Vector3s pos;
	/** Offset in the excess list. */
	int offset;
	/** Pointer to the voxel block array.
		- >= 0 identifies an actual allocated entry in the voxel block array
		- -1 identifies an entry that has been removed (swapped out)
		- <-1 identifies an unallocated block
	*/
	int ptr;

	/** Corresponding TSDFBase direction. */
	uint8_t direction;

	_CPU_AND_GPU_CODE_ bool IsValid() const {
		return ptr >= 0;
	}
};

namespace ITMLib
{

	_CPU_AND_GPU_CODE_
	inline int hash(int seed, int value)
	{
		return (seed * value) % 16235657;
	}

	/**
	 * l-word hash function as suggested by M. N. Wegman 1977
	 *
	 * h(xyz) = h1(x) + h2(y) + h3(z) mod M
	 * @tparam T
	 * @param blockPos
	 * @return
	 */
	template<typename T> _CPU_AND_GPU_CODE_
	inline int hashIndex(const THREADPTR(T) & blockPos) {
		return (hash(11536487, blockPos.x)
						+ hash(14606887, blockPos.y)
						+ hash(28491781, blockPos.z)) % (uint)SDF_HASH_MASK;
	}

	template<typename T> _CPU_AND_GPU_CODE_
	inline int hashIndex(const THREADPTR(T) & blockPos, const TSDFDirection direction) {
		if (direction == TSDFDirection::NONE)
			return hashIndex(blockPos);
		return (hash(11536487, blockPos.x)
		        + hash(14606887, blockPos.y)
		        + hash(28491781, blockPos.z)
		        + hash(83492791, static_cast<TSDFDirection_type>(direction))) % (uint)SDF_HASH_MASK;
	}

	/** \brief
	This is the central class for the voxel block hash
	implementation. It contains all the data needed on the CPU
	and a pointer to the data structure on the GPU.
	*/
	class ITMVoxelBlockHash
	{
	public:
		typedef ITMHashEntry IndexData;

		typedef struct _IndexCache {
			Vector3i blockPos;
			int blockPtr;
			_CPU_AND_GPU_CODE_ _IndexCache(void) : blockPos(0x7fffffff), blockPtr(-1) {}
		} IndexCache ;

		/** Maximum number of total entries. */
		static const int noTotalEntries = SDF_BUCKET_NUM + SDF_EXCESS_LIST_SIZE;
		static const int noLocalEntries = SDF_LOCAL_BLOCK_NUM;
		static const int voxelBlockSize = SDF_BLOCK_SIZE3;

	private:
		int lastFreeExcessListId;

		/** The actual data in the hash table. */
		ORUtils::MemoryBlock<ITMHashEntry> *hashEntries;

		/** Identifies which entries of the overflow
		list are allocated. This is used if too
		many hash collisions caused the buckets to
		overflow.
		*/
		ORUtils::MemoryBlock<int> *excessAllocationList;

		MemoryDeviceType memoryType;

	public:
		ITMVoxelBlockHash(MemoryDeviceType memoryType)
		{
			this->memoryType = memoryType;
			hashEntries = new ORUtils::MemoryBlock<ITMHashEntry>(noTotalEntries, memoryType);
			excessAllocationList = new ORUtils::MemoryBlock<int>(SDF_EXCESS_LIST_SIZE, memoryType);
		}

		~ITMVoxelBlockHash(void)
		{
			delete hashEntries;
			delete excessAllocationList;
		}

		void Reset()
		{
			hashEntries->Clear();
			excessAllocationList->Clear();
		}

		/** Get the list of actual entries in the hash table. */
		const ITMHashEntry *GetEntries(void) const { return hashEntries->GetData(memoryType); }
		ITMHashEntry *GetEntries(void) { return hashEntries->GetData(memoryType); }

		const IndexData *getIndexData(void) const { return hashEntries->GetData(memoryType); }
		IndexData *getIndexData(void) { return hashEntries->GetData(memoryType); }

		/** Get the list that identifies which entries of the
		overflow list are allocated. This is used if too
		many hash collisions caused the buckets to overflow.
		*/
		const int *GetExcessAllocationList(void) const { return excessAllocationList->GetData(memoryType); }
		int *GetExcessAllocationList(void) { return excessAllocationList->GetData(memoryType); }

		int GetLastFreeExcessListId(void) { return lastFreeExcessListId; }
		void SetLastFreeExcessListId(int lastFreeExcessListId) { this->lastFreeExcessListId = lastFreeExcessListId; }

		/** Maximum number of total entries. */
		int getNumAllocatedVoxelBlocks(void) { return noLocalEntries; }
		int getVoxelBlockSize(void) { return voxelBlockSize; }

		void SaveToDirectory(const std::string &outputDirectory) const
		{
			std::string hashEntriesFileName = outputDirectory + "hash.dat";
			std::string excessAllocationListFileName = outputDirectory + "excess.dat";
			std::string lastFreeExcessListIdFileName = outputDirectory + "last.txt";

			std::ofstream ofs(lastFreeExcessListIdFileName.c_str());
			if (!ofs) throw std::runtime_error("Could not open " + lastFreeExcessListIdFileName + " for writing");

			ofs << lastFreeExcessListId;
			ORUtils::MemoryBlockPersister::SaveMemoryBlock(hashEntriesFileName, *hashEntries, memoryType);
			ORUtils::MemoryBlockPersister::SaveMemoryBlock(excessAllocationListFileName, *excessAllocationList, memoryType);
		}

		void LoadFromDirectory(const std::string &inputDirectory)
		{
			std::string hashEntriesFileName = inputDirectory + "hash.dat";
			std::string excessAllocationListFileName = inputDirectory + "excess.dat";
			std::string lastFreeExcessListIdFileName = inputDirectory + "last.txt";

			std::ifstream ifs(lastFreeExcessListIdFileName.c_str());
			if (!ifs) throw std::runtime_error("Count not open " + lastFreeExcessListIdFileName + " for reading");

			ifs >> this->lastFreeExcessListId;
			ORUtils::MemoryBlockPersister::LoadMemoryBlock(hashEntriesFileName.c_str(), *hashEntries, memoryType);
			ORUtils::MemoryBlockPersister::LoadMemoryBlock(excessAllocationListFileName.c_str(), *excessAllocationList, memoryType);
		}

		// Suppress the default copy constructor and assignment operator
		ITMVoxelBlockHash(const ITMVoxelBlockHash&);
		ITMVoxelBlockHash& operator=(const ITMVoxelBlockHash&);
	};

	/**
	 * Type for representing whether a HashEntry requires allocation
	 */
	enum HashEntryAllocType : uchar
	{
		ALLOCATED = 0,
		ALLOCATE_ORDERED = 1, // entry requires allocation in ordered list
		ALLOCATE_EXCESS = 2  // entry requires allocation in excess list
	};

/**
 * Type for representing whether a block corresponding to a HashEntry is visible and in memory
 */
	enum HashEntryVisibilityType : uchar
	{
		INVISIBLE = 0,
		VISIBLE_IN_MEMORY = 1,
		VISIBLE_STREAMED_OUT = 2,
		PREVIOUSLY_VISIBLE = 3 // visible at previous frame and unstreamed
	};

} // namespace ITMLib
