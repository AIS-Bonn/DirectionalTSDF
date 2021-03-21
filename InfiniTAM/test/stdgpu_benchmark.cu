//
// Created by Malte Splietker on 20.03.21.
//

#include <ORUtils/SE3Pose.h>
#include <ITMLib/Utils/ITMCUDAUtils.h>
#include <ITMLib/Utils/ITMMath.h>
#include <ITMLib/ITMLibDefines.h>
#include <ITMLib/Utils/ITMTimer.h>
#include <ITMLib/Objects/Scene/ITMRepresentationAccess.h>
#include <stdgpu/unordered_map.cuh>

using namespace ITMLib;

typedef stdgpu::unordered_map<Vector3s, ITMVoxel*> HashMapStdGpu;

static const int RUNS = 100;

__global__ void addBlocksITM(ITMHashEntry* hashTable, int* noAllocatedVoxelEntries)
{
	Vector3s blockPos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y,
	                  blockIdx.z * blockDim.z + threadIdx.z);

	int vbaIdx = atomicSub(noAllocatedVoxelEntries, 1);
	int targetIdx = hashIndex(blockPos);

	if (vbaIdx >= 0) //there is room in the voxel block array
	{
		ITMHashEntry hashEntry;
		hashEntry.pos.x = blockPos.x; hashEntry.pos.y = blockPos.y; hashEntry.pos.z = blockPos.z;
		hashEntry.ptr = 0;
		hashEntry.offset = 0;

		hashTable[targetIdx] = hashEntry;
	}
	else
	{
		atomicAdd(noAllocatedVoxelEntries, 1);
	}
}

__global__ void addBlocksStdGpu(HashMapStdGpu map)
{
	Vector3s blockPos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y,
	                  blockIdx.z * blockDim.z + threadIdx.z);

	map.emplace(blockPos, nullptr);
}

__global__ void serialAccessITM(ITMHashEntry* hashTable, int* noMisses)
{
	Vector3s blockPos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y,
	                  blockIdx.z * blockDim.z + threadIdx.z);

	int hashIdx = hashIndex(blockPos);

	bool found = false;
	ITMHashEntry hashEntry;
	hashEntry.ptr = -1;
	while (true)
	{
		hashEntry = hashTable[hashIdx];

		if (IS_EQUAL3(hashEntry.pos, blockPos) && hashEntry.ptr >= 0)
		{
			found = true;
			break;
		}

		if (hashEntry.offset < 1) break;
		hashIdx = SDF_BUCKET_NUM + hashEntry.offset - 1;
	}

	if (not found)
		atomicAdd(noMisses, 1);
}

__global__ void serialAccessStdGpu(const HashMapStdGpu map, int* noMisses)
{
	Vector3s blockPos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y,
	                  blockIdx.z * blockDim.z + threadIdx.z);

	auto it = map.find(blockPos);
	if (it == map.end())
	{
		atomicAdd(noMisses, 1);
	}
}

__global__ void randomAccessITM(ITMHashEntry* hashTable, int* noMisses)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	Vector3s blockPos(abs(hash(11536487, i)) % (SDF_LOCAL_BLOCK_NUM / (8 * 8 * 8)),
	                  abs(hash(14606887, i)) % 8,
	                  abs(hash(28491781, i)) % 8);

	int hashIdx = hashIndex(blockPos);

	bool found = false;
	ITMHashEntry hashEntry;
	hashEntry.ptr = -1;
	while (true)
	{
		hashEntry = hashTable[hashIdx];

		if (IS_EQUAL3(hashEntry.pos, blockPos) && hashEntry.ptr >= 0)
		{
			found = true;
			break;
		}

		if (hashEntry.offset < 1) break;
		hashIdx = SDF_BUCKET_NUM + hashEntry.offset - 1;
	}

	if (not found)
		atomicAdd(noMisses, 1);
}

__global__ void randomAccessStdGpu(const HashMapStdGpu map, int* noMisses)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	Vector3s blockPos(abs(hash(11536487, i)) % (SDF_LOCAL_BLOCK_NUM / (8 * 8 * 8)),
                    abs(hash(14606887, i)) % 8,
                    abs(hash(28491781, i)) % 8);

	auto it = map.find(blockPos);
	if (it == map.end())
	{
		atomicAdd(noMisses, 1);
	}
}

int main()
{
	HashMapStdGpu map = HashMapStdGpu::createDeviceObject(10000000);

	ITMVoxelIndex* hashMap = new ITMVoxelBlockHash(MEMORYDEVICE_CUDA);
	ITMLocalVBA<ITMVoxel>* voxels = new ITMLocalVBA<ITMVoxel>(MEMORYDEVICE_CUDA, hashMap->getNumAllocatedVoxelBlocks(),
	                                                          hashMap->getVoxelBlockSize());

	{
		dim3 blockSize(8, 8, 8);
		dim3 gridSize(hashMap->getNumAllocatedVoxelBlocks() / (blockSize.x * blockSize.y * blockSize.z));

		printf("Adding %i entries to hash table\n",
		       blockSize.x * blockSize.y * blockSize.z * gridSize.x * gridSize.y * gridSize.z);
		ITMTimer timer;
		float sum = 0;
		for (int i = 0; i < RUNS; i++)
		{
			hashMap->Reset();
			timer.Tick();
			int* noAllocatedVoxelEntries;
			int n = hashMap->getNumAllocatedVoxelBlocks();
			ORcudaSafeCall(cudaMalloc(&noAllocatedVoxelEntries, sizeof(int)));
			ORcudaSafeCall(cudaMemcpy(noAllocatedVoxelEntries, &n, sizeof(int), cudaMemcpyHostToDevice));
			addBlocksITM<<<gridSize, blockSize>>>(hashMap->GetEntries(), noAllocatedVoxelEntries);
			cudaDeviceSynchronize();
			ORcudaKernelCheck
			cudaFree(noAllocatedVoxelEntries);
			sum += timer.Tock() / RUNS;
		}
		printf("\tITMLib: %f\t\t(NO COLLISION HANDLING!)\n", sum);

		sum = 0;
		for (int i = 0; i < RUNS; i++)
		{
			map.clear();
			timer.Tick();
			addBlocksStdGpu<<<gridSize, blockSize>>>(map);
			ORcudaKernelCheck
			sum += timer.Tock() / RUNS;
		}
		printf("\tstdgpu: %f\n", sum);
	}

	{
		dim3 blockSize(8, 8, 8);
		dim3 gridSize(hashMap->getNumAllocatedVoxelBlocks() / (blockSize.x * blockSize.y * blockSize.z));

		printf("Serial access of %i entries\n", blockSize.x * blockSize.y * blockSize.z * gridSize.x * gridSize.y * gridSize.z);

		ITMTimer timer;
		float sum = 0;
		for (int i = 0; i < RUNS; i++)
		{
			timer.Tick();
			int* noMisses_device;
			ORcudaSafeCall(cudaMalloc(&noMisses_device, sizeof(int)));
			ORcudaSafeCall(cudaMemset(noMisses_device, 0, 0));
			serialAccessITM<<<gridSize, blockSize>>>(hashMap->GetEntries(), noMisses_device);
			int noMisses;
			ORcudaSafeCall(cudaMemcpy(&noMisses, noMisses_device, sizeof(int), cudaMemcpyDeviceToHost));
//			printf("misses: %i\n", noMisses);
			ORcudaKernelCheck
			sum += timer.Tock() / RUNS;
		}
		printf("\tITMLib: %f\n", sum);

		sum = 0;
		for (int i = 0; i < RUNS; i++)
		{
			timer.Tick();
			int* noMisses_device;
			ORcudaSafeCall(cudaMalloc(&noMisses_device, sizeof(int)));
			ORcudaSafeCall(cudaMemset(noMisses_device, 0, 0));
			serialAccessStdGpu<<<gridSize, blockSize>>>(map, noMisses_device);
			int noMisses;
			ORcudaSafeCall(cudaMemcpy(&noMisses, noMisses_device, sizeof(int), cudaMemcpyDeviceToHost));
//			printf("misses: %i\n", noMisses);
			ORcudaKernelCheck
			sum += timer.Tock() / RUNS;
		}
		printf("\tstdgpu: %f\n", sum);
	}

	{
		dim3 blockSize(512);
		dim3 gridSize(1000000 / 512);

		printf("Random access of %i entries\n", blockSize.x * blockSize.y * blockSize.z * gridSize.x * gridSize.y * gridSize.z);

		ITMTimer timer;
		float sum = 0;
		for (int i = 0; i < RUNS; i++)
		{
			timer.Tick();
			int* noMisses_device;
			ORcudaSafeCall(cudaMalloc(&noMisses_device, sizeof(int)));
			ORcudaSafeCall(cudaMemset(noMisses_device, 0, 0));
			randomAccessITM<<<gridSize, blockSize>>>(hashMap->GetEntries(), noMisses_device);
			int noMisses;
			ORcudaSafeCall(cudaMemcpy(&noMisses, noMisses_device, sizeof(int), cudaMemcpyDeviceToHost));
//			printf("misses: %i\n", noMisses);
			ORcudaKernelCheck
			sum += timer.Tock() / RUNS;
		}
		printf("\tITMLib: %f\n", sum);

		sum = 0;
		for (int i = 0; i < RUNS; i++)
		{
			timer.Tick();
			int* noMisses_device;
			ORcudaSafeCall(cudaMalloc(&noMisses_device, sizeof(int)));
			ORcudaSafeCall(cudaMemset(noMisses_device, 0, 0));
			randomAccessStdGpu<<<gridSize, blockSize>>>(map, noMisses_device);
			int noMisses;
			ORcudaSafeCall(cudaMemcpy(&noMisses, noMisses_device, sizeof(int), cudaMemcpyDeviceToHost));
//			printf("misses: %i\n", noMisses);
			ORcudaKernelCheck
			sum += timer.Tock() / RUNS;
		}
		printf("\tstdgpu: %f\n", sum);
	}

	HashMapStdGpu::destroyDeviceObject(map);
	delete hashMap;
	delete voxels;

	return 0;
}