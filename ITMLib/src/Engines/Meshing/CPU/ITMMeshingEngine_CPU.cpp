// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMMeshingEngine_CPU.h"

#include <unordered_set>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <ITMLib/ITMLibDefines.h>
#include <Objects/TSDF_CPU.h>
#include "../Shared/ITMMeshingEngine_Shared.h"

using namespace ITMLib;

/**
 * find all allocated blocks (xyz only), dropping direction component
 * @tparam TIndex
 * @param tsdf
 * @param allBlocksList
 * @param numberBlocks
 */
template<typename TIndex>
void
findAllocatedBlocks(const std::unordered_map<TIndex, ITMVoxel*>& tsdf, ITMIndex** allBlocksList, size_t& numberBlocks)
{
	std::unordered_set<ITMIndex> allBlocks;
	allBlocks.reserve(tsdf.size());

	thrust::for_each(thrust::host, tsdf.begin(), tsdf.end(),
	                 findAllocatedBlocksFunctor<TIndex, std::unordered_set>(allBlocks));

	*allBlocksList = (ITMIndex*) malloc(allBlocks.size() * sizeof(ITMIndex));
	thrust::copy(allBlocks.begin(), allBlocks.end(), *allBlocksList);
	numberBlocks = allBlocks.size();
}

void ITMMeshingEngine_CPU::MeshScene(ITMMesh* mesh, const Scene* scene)
{
	ITMIndex* allBlocksList;
	size_t numberBlocks;

	auto tsdf = scene->tsdf->toCPU()->getMap();
	findAllocatedBlocks(tsdf, &allBlocksList, numberBlocks);
	printf("found %zu blocks ... ", numberBlocks);


	size_t noTriangles = 0;
	mesh->Resize(numberBlocks * SDF_BLOCK_SIZE3 * 4);
	ITMMesh::Triangle* triangles = mesh->triangles->GetData(MEMORYDEVICE_CPU);
	float factor = scene->sceneParams->voxelSize;

	for (size_t entryId = 0; entryId < numberBlocks; entryId++)
	{
		ITMIndex block = allBlocksList[entryId];

		Vector3i globalPos = Vector3i(block.x, block.y, block.z) * SDF_BLOCK_SIZE;

		Vector3f vertList[12];
		for (int x = 0; x < SDF_BLOCK_SIZE; x++)
			for (int y = 0; y < SDF_BLOCK_SIZE; y++)
				for (int z = 0; z < SDF_BLOCK_SIZE; z++)
				{
					int cubeIndex = buildVertList(vertList, globalPos, Vector3i(x, y, z), tsdf);

					if (cubeIndex < 0) continue;

					for (int i = 0; triangleTable[cubeIndex][i] != -1; i += 3)
					{
#ifdef WITH_OPENMP
#pragma omp atomic capture
#endif
						size_t triangleId = noTriangles++;

						if (triangleId < mesh->noMaxTriangles - 1)
						{
							triangles[triangleId].p0 = vertList[triangleTable[cubeIndex][i]] * factor;
							triangles[triangleId].p1 = vertList[triangleTable[cubeIndex][i + 1]] * factor;
							triangles[triangleId].p2 = vertList[triangleTable[cubeIndex][i + 2]] * factor;
						}
					}
				}
	}

	mesh->noTotalTriangles = MIN(mesh->noMaxTriangles, noTriangles);
}