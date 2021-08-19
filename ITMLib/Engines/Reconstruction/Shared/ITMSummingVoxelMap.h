//
// Created by Malte Splietker on 05.02.21.
//

#pragma once

#include <ITMLib/Objects/Scene/ITMVoxelBlockHash.h>
#include "ITMLib/Objects/Scene/ITMSummingVoxel.h"

namespace ITMLib
{

typedef Vector4s IndexType;

/**
 * Abstract interface for map for summing voxels.
 * in Init() the map is to be initialized (and entries reset) with the given visible entries
 * @tparam Map map type. Either std::unordered_map or stdgpu::unordered_map
 */
template<template <typename, typename...> class Map>
class SummingVoxelMap
{
public:
	virtual ~SummingVoxelMap() = default;

	virtual void Init(const ITMVoxelBlockHash::IndexData* hashTable, const int* visibleEntryIds, int noVisibleEntries) = 0;

	inline Map<IndexType, SummingVoxel*>& getMap() { return map; }

	inline SummingVoxel* getVoxels() { return summingVoxels; }

protected:
	SummingVoxel* summingVoxels = nullptr;
	Map<IndexType, SummingVoxel*> map;
};

} // namespace ITMLib
