// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMMeshingEngine_CUDA.h"

#include <algorithm>
#include <atomic>
#include <ITMLib/ITMLibDefines.h>
#include <Objects/TSDF_CUDA.h>
#include <Utils/ITMCUDAUtils.h>

#include "../Shared/ITMMeshingEngine_Shared.h"

#include <ORUtils/CUDADefines.h>

#include <stdgpu/unordered_map.cuh>
#include <stdgpu/unordered_set.cuh>

using namespace ITMLib;

template<typename TIndex>
__global__ void
meshScene_device(ITMMesh::Triangle* triangles, ITMMesh::TriangleColor* triangleColors, unsigned int* noTriangles_device,
                 float voxelSize, int noMaxTriangles, const ITMIndex* blocksList,
                 stdgpu::unordered_map<TIndex, ITMVoxel*> tsdf,
                 unsigned int offset = 0, unsigned int lastBlock = 0);

ITMMeshingEngine_CUDA::ITMMeshingEngine_CUDA()
{
	ORcudaSafeCall(cudaMalloc((void**) &noTriangles_device, sizeof(unsigned int)));
}

ITMMeshingEngine_CUDA::~ITMMeshingEngine_CUDA()
{
	ORcudaSafeCall(cudaFree(noTriangles_device));
}

/**
 * find all allocated blocks (xyz only), dropping direction component
 * @tparam TIndex
 * @param tsdf
 * @param allBlocksList
 * @param numberBlocks
 */
template<typename TIndex>
void
findAllocatedBlocks(const stdgpu::unordered_map<TIndex, ITMVoxel*> tsdf, ITMIndex** allBlocksList, size_t& numberBlocks)
{
	stdgpu::unordered_set<ITMIndex> allBlocks = stdgpu::unordered_set<ITMIndex>::createDeviceObject(tsdf.size());
	thrust::for_each(thrust::device, tsdf.device_range().begin(), tsdf.device_range().end(),
	                 findAllocatedBlocksFunctor<TIndex, stdgpu::unordered_set>(allBlocks));

	ORcudaSafeCall(cudaMalloc(allBlocksList, allBlocks.size() * sizeof(ITMIndex)));
	thrust::copy(allBlocks.device_range().begin(), allBlocks.device_range().end(), stdgpu::device_begin(*allBlocksList));
	numberBlocks = allBlocks.size();

	stdgpu::unordered_set<ITMIndex>::destroyDeviceObject(allBlocks);
}

void ITMMeshingEngine_CUDA::MeshScene(ITMMesh* mesh, const Scene* scene)
{
//	MeshSceneDefault(mesh, scene);
	MeshSceneStreamed(mesh, scene);
}

// Deprecated, not maintained anymore
void ITMMeshingEngine_CUDA::MeshSceneDefault(ITMMesh* mesh, const Scene* scene)
{
	ORcudaSafeCall(cudaMemset(noTriangles_device, 0, sizeof(unsigned int)));

	ITMIndex* allBlocksList;
	size_t numberBlocks;

	auto tsdf = scene->tsdf->toCUDA()->getMap();
	findAllocatedBlocks(tsdf, &allBlocksList, numberBlocks);
	printf("found %zu blocks ... ", numberBlocks);

	dim3 cudaBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
	dim3 gridSize(numberBlocks);

	ITMMesh::Triangle* triangles_device;
	size_t noMaxTriangles = numberBlocks * SDF_BLOCK_SIZE3 * 4;
	ORcudaSafeCall(cudaMalloc(&triangles_device, sizeof(ITMMesh::Triangle) * noMaxTriangles));

	meshScene_device << < gridSize, cudaBlockSize >> >(triangles_device, nullptr, noTriangles_device,
		scene->sceneParams->voxelSize, noMaxTriangles,
		allBlocksList, tsdf);
	ORcudaKernelCheck;

	ORcudaSafeCall(cudaMemcpy(&mesh->noTotalTriangles, noTriangles_device, sizeof(unsigned int), cudaMemcpyDeviceToHost));

	// meshScene_device made sure up to noMaxTriangles triangles were copied in the output array but,
	// since the check was performed with atomicAdds, the actual number stored in noTriangles_device
	// might be greater than noMaxTriangles.
	// We coerce it to be lesser or equal to that number, not doing it causes a segfault when using the mesh later.
	mesh->noTotalTriangles = std::min<uint>(mesh->noTotalTriangles, static_cast<uint>(noMaxTriangles));
	mesh->Resize(mesh->noTotalTriangles);

	ORcudaSafeCall(cudaMemcpy(mesh->triangles->GetData(MEMORYDEVICE_CPU), triangles_device,
	                          sizeof(ITMMesh::Triangle) * mesh->noTotalTriangles, cudaMemcpyDeviceToHost));

	ORcudaSafeCall(cudaFree(triangles_device));
	ORcudaSafeCall(cudaFree(allBlocksList));
}

void ITMMeshingEngine_CUDA::MeshSceneStreamed(ITMMesh* mesh, const Scene* scene)
{
	ORcudaSafeCall(cudaMemset(noTriangles_device, 0, sizeof(unsigned int)));

	ITMIndex* allBlocksList;
	size_t numberBlocks;

	auto tsdf = scene->tsdf->toCUDA()->getMap();
	findAllocatedBlocks(tsdf, &allBlocksList, numberBlocks);
	printf("found %zu blocks ... ", numberBlocks);

	const int blocksPerStream = 128;
	const int nStreams = 4;
	const int N = std::ceil(static_cast<float>(numberBlocks) / (blocksPerStream * nStreams));
	const size_t maxNoTriangles = blocksPerStream * SDF_BLOCK_SIZE3 * 4;

	cudaStream_t stream[nStreams];
	ITMMesh::Triangle* triangles_device[nStreams];
	ITMMesh::TriangleColor* triangleColors_device[nStreams];
	unsigned int* noTriangles_device[nStreams];

	mesh->noTotalTriangles = 0;
	mesh->Resize(numberBlocks * SDF_BLOCK_SIZE3 * 4);

	for (int i = 0; i < nStreams; i++)
	{
		ORcudaSafeCall(cudaStreamCreate(&stream[i]));
		ORcudaSafeCall(cudaMalloc(&noTriangles_device[i], sizeof(unsigned int)));
		ORcudaSafeCall(cudaMalloc(&triangles_device[i], sizeof(ITMMesh::Triangle) * maxNoTriangles));
		triangleColors_device[i] = nullptr;
		if (mesh->withColor)
			ORcudaSafeCall(cudaMalloc(&triangleColors_device[i], sizeof(ITMMesh::TriangleColor) * maxNoTriangles));
	}
	for (int j = 0; j < N; j++)
		for (int i = 0; i < nStreams; i++)
		{
			ORcudaSafeCall(cudaMemsetAsync(noTriangles_device[i], 0, sizeof(unsigned int), stream[i]));

			int offset = (j * nStreams + i) * blocksPerStream;

			dim3 cudaBlockSize(SDF_BLOCK_SIZE, SDF_BLOCK_SIZE, SDF_BLOCK_SIZE);
			meshScene_device << < blocksPerStream, cudaBlockSize, 0, stream[i] >> >(
				triangles_device[i], triangleColors_device[i], noTriangles_device[i],
					scene->sceneParams->voxelSize,
					maxNoTriangles, allBlocksList, tsdf,
					offset, numberBlocks);

			unsigned int noTriangles_cpu;
			ORcudaSafeCall(
				cudaMemcpyAsync(&noTriangles_cpu, noTriangles_device[i], sizeof(unsigned int), cudaMemcpyDeviceToHost,
				                stream[i]));

			ORcudaSafeCall(cudaMemcpyAsync(&mesh->triangles->GetData(MEMORYDEVICE_CPU)[mesh->noTotalTriangles],
			                               triangles_device[i], sizeof(ITMMesh::Triangle) * noTriangles_cpu,
			                               cudaMemcpyDeviceToHost, stream[i]));
			if (mesh->withColor)
				ORcudaSafeCall(cudaMemcpyAsync(&mesh->triangleColors->GetData(MEMORYDEVICE_CPU)[mesh->noTotalTriangles],
				                               triangleColors_device[i], sizeof(ITMMesh::TriangleColor) * noTriangles_cpu,
				                               cudaMemcpyDeviceToHost, stream[i]));
			mesh->noTotalTriangles += noTriangles_cpu;
		}
	for (int i = 0; i < nStreams; i++)
	{
		ORcudaSafeCall(cudaFree(triangles_device[i]));
		ORcudaSafeCall(cudaFree(noTriangles_device[i]));
		ORcudaSafeCall(cudaStreamDestroy(stream[i]));
	}

	ORcudaSafeCall(cudaFree(allBlocksList));
}

template<typename TIndex>
__global__ void
meshScene_device(ITMMesh::Triangle* triangles, ITMMesh::TriangleColor* triangleColors, unsigned int* noTriangles_device,
                 float voxelSize, int noMaxTriangles, const ITMIndex* blocksList,
                 const stdgpu::unordered_map<TIndex, ITMVoxel*> tsdf,
                 unsigned int offset, unsigned int lastBlock)
{
	const ITMIndex block = blocksList[blockIdx.x + gridDim.x * blockIdx.y + offset];

	if (lastBlock > 0 and blockIdx.x + gridDim.x * blockIdx.y + offset > lastBlock)
		return;

	Vector3i globalPos = Vector3i(block.x, block.y, block.z) * SDF_BLOCK_SIZE;

	Vector3f vertList[12];
	int cubeIndex = buildVertList(vertList, globalPos, Vector3i(threadIdx.x, threadIdx.y, threadIdx.z), tsdf);

	if (cubeIndex < 0) return;

	for (int i = 0; triangleTable[cubeIndex][i] != -1; i += 3)
	{
		int triangleId = atomicAdd(noTriangles_device, 1);

		if (triangleId < noMaxTriangles - 1)
		{
			triangles[triangleId].p0 = vertList[triangleTable[cubeIndex][i]] * voxelSize;
			triangles[triangleId].p1 = vertList[triangleTable[cubeIndex][i + 1]] * voxelSize;
			triangles[triangleId].p2 = vertList[triangleTable[cubeIndex][i + 2]] * voxelSize;
			if (triangleColors)
			{
				triangleColors[triangleId].p0 = (255 * readFromSDF_color4u_interpolated(tsdf,
				                                                                        vertList[triangleTable[cubeIndex][i]]).toVector3()).toUChar();
				triangleColors[triangleId].p1 = (255 * readFromSDF_color4u_interpolated(tsdf,
				                                                                        vertList[triangleTable[cubeIndex][i +
				                                                                                                          1]]).toVector3()).toUChar();
				triangleColors[triangleId].p2 = (255 * readFromSDF_color4u_interpolated(tsdf,
				                                                                        vertList[triangleTable[cubeIndex][i +
				                                                                                                          2]]).toVector3()).toUChar();
			}
		}
	}
}
