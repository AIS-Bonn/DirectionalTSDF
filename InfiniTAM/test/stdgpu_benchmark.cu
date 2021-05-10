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

__global__ void addBlocksITM(ITMHashEntry* hashTable, ITMVoxel* voxels, int* noAllocatedVoxelEntries)
{
	Vector3s blockPos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y,
	                  blockIdx.z * blockDim.z + threadIdx.z);

	int vbaIdx = atomicSub(noAllocatedVoxelEntries, 1);
	int targetIdx = hashIndex(blockPos);

	if (vbaIdx >= 0) //there is room in the voxel block array
	{
		ITMHashEntry hashEntry;
		hashEntry.pos.x = blockPos.x; hashEntry.pos.y = blockPos.y; hashEntry.pos.z = blockPos.z;
		hashEntry.ptr = vbaIdx;
		hashEntry.offset = 0;

		hashTable[targetIdx] = hashEntry;
	}
	else
	{
		atomicAdd(noAllocatedVoxelEntries, 1);
	}
}

__global__ void addBlocksStdGpu(HashMapStdGpu map, ITMVoxel* voxels, int* counter)
{
	Vector3s blockPos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y,
	                  blockIdx.z * blockDim.z + threadIdx.z);

	int id = atomicAdd(counter, 1);
	map.emplace(blockPos, voxels + id * SDF_BLOCK_SIZE3);
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

__global__ void readUninterpolatedFromTSDFITM(ITMHashEntry* hashTable, ITMVoxel* voxels, float* sum)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	Vector3i voxelIdx(abs(hash(11536487, i)) % (SDF_LOCAL_BLOCK_NUM / (8 * 8)),
	                  abs(hash(14606887, i)) % (8 * 8),
	                  abs(hash(28491781, i)) % (8 * 8));

	ITMVoxelBlockHash::IndexCache cache;
	int vmIndex;
	*sum += readFromSDF_float_uninterpolated(voxels, hashTable, voxelIdx.toFloat(), TSDFDirection::NONE, vmIndex, cache);
}

__global__ void readUninterpolatedFromTSDFStdGpu(const HashMapStdGpu map, float* sum)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	Vector3i voxelIdx(abs(hash(11536487, i)) % (SDF_LOCAL_BLOCK_NUM / (8 * 8)),
	                  abs(hash(14606887, i)) % (8 * 8),
	                  abs(hash(28491781, i)) % (8 * 8));

	bool found;
	*sum += readFromSDF_float_uninterpolated(found, map, voxelIdx.toFloat());
}

__global__ void readInterpolatedFromTSDFITM(ITMHashEntry* hashTable, ITMVoxel* voxels, float* sum)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	Vector3i voxelIdx(abs(hash(11536487, i)) % (SDF_LOCAL_BLOCK_NUM / (8 * 8)),
	                  abs(hash(14606887, i)) % (8 * 8),
	                  abs(hash(28491781, i)) % (8 * 8));

	ITMVoxelBlockHash::IndexCache cache;
	int vmIndex;
	*sum += readFromSDF_float_interpolated(voxels, hashTable, voxelIdx.toFloat(), TSDFDirection::NONE, vmIndex, cache);
//	float confidence;
//	*sum += readWithConfidenceFromSDF_float_interpolated(confidence, voxels, hashTable, voxelIdx.toFloat(), TSDFDirection::NONE, 100, vmIndex, cache);
}

__global__ void readInterpolatedFromTSDFStdGpu(const HashMapStdGpu map, float* sum)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	Vector3i voxelIdx(abs(hash(11536487, i)) % (SDF_LOCAL_BLOCK_NUM / (8 * 8)),
	                  abs(hash(14606887, i)) % (8 * 8),
	                  abs(hash(28491781, i)) % (8 * 8));

	bool found;
	*sum += readFromSDF_float_interpolated(found, map, voxelIdx.toFloat());
//	float confidence;
//	*sum += readWithConfidenceFromSDF_float_interpolated(found, confidence, map, voxelIdx.toFloat(), 100);
}

__global__ void rayCastITM(ITMHashEntry* hashTable, ITMVoxel* voxels, float* sum, int* misses)
{
	int x = (threadIdx.x), y = (threadIdx.y);
	if (x >= 640 || y >= 480) return;

	int vmIndex;
	ITMVoxelBlockHash::IndexCache cache;
	Vector3f pt(blockIdx.x + blockIdx.y, x, y);
	for (int i = 0; i < 10; i++)
	{
		*sum += readFromSDF_float_uninterpolated(voxels, hashTable, pt, TSDFDirection::NONE, vmIndex, cache);
		if (not vmIndex)
			atomicAdd(misses, 1);
		pt.x += SDF_BLOCK_SIZE;
	}

	*sum += readFromSDF_float_interpolated(voxels, hashTable, pt, TSDFDirection::NONE, vmIndex, cache);
	if (not vmIndex)
		atomicAdd(misses, 1);
	float confidence;
	for (int i = 0; i < 10; i++)
	{
		pt.x += 0.5 - (i % 2);
		*sum += readWithConfidenceFromSDF_float_interpolated(confidence, voxels, hashTable, pt, TSDFDirection::NONE, 100, vmIndex, cache);
		if (not vmIndex)
			atomicAdd(misses, 1);
	}
	*sum += confidence;
}

__global__ void rayCastStdGpu(const HashMapStdGpu map, float* sum, int* misses)
{
	int x = (threadIdx.x), y = (threadIdx.y);
	if (x >= 640 || y >= 480) return;

	bool found;
	Vector3f pt(blockIdx.x + blockIdx.y, x, y);
	for (int i = 0; i < 10; i++)
	{
		*sum += readFromSDF_float_uninterpolated(found, map, pt);
		if (not found)
			atomicAdd(misses, 1);
		pt.x += SDF_BLOCK_SIZE;
	}

	*sum += readFromSDF_float_interpolated(found, map, pt);
	if (not found)
		atomicAdd(misses, 1);

	float confidence;
	for (int i = 0; i < 10; i++)
	{
		pt.x += 0.5 - (i % 2);
		*sum += readWithConfidenceFromSDF_float_interpolated(found, confidence, map, pt, 100);
		if (not found)
			atomicAdd(misses, 1);
	}
	*sum += confidence;
}

__global__ void eraseMissingBlocksFromStdGpu(HashMapStdGpu map, const ITMHashEntry* hashTable, const ITMVoxel* voxels)
{
	Vector3i blockPos(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y,
	                  blockIdx.z * blockDim.z + threadIdx.z);

	ITMHashEntry entry = getHashEntry(hashTable, blockPos);

	if (not entry.IsValid())
		map.erase(blockPos.toShort());
}

int main()
{
	HashMapStdGpu map = HashMapStdGpu::createDeviceObject(10000000);

	ITMVoxelIndex* hashMap = new ITMVoxelBlockHash(MEMORYDEVICE_CUDA);
	ITMVoxel v;
	v.sdf = 1;
	v.w_depth = 1;
	ITMVoxel* voxels = createDeviceArray(hashMap->getNumAllocatedVoxelBlocks() * hashMap->getVoxelBlockSize(), v);

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
			addBlocksITM<<<gridSize, blockSize>>>(hashMap->GetEntries(), voxels, noAllocatedVoxelEntries);
			cudaDeviceSynchronize();
			ORcudaKernelCheck
			cudaFree(noAllocatedVoxelEntries);
			sum += timer.Tock() / RUNS;
		}
		float ratio = 1 / sum;
		printf("\tITMLib: %f\t\t(NO COLLISION HANDLING!)\n", sum);

		sum = 0;
		for (int i = 0; i < RUNS; i++)
		{
			map.clear();
			timer.Tick();
			int* counter = createDeviceArray<int>(1, 0);
			addBlocksStdGpu<<<gridSize, blockSize>>>(map, voxels, counter);
			ORcudaKernelCheck
			sum += timer.Tock() / RUNS;
		}
		ratio *= sum;
		printf("\tstdgpu: %f\n", sum);
		printf("\tratio: %.2f\n", ratio);
	}

	// DEBUG: remove blocks which are missing from ITM hashmap, to get comparable readings
	{
		dim3 blockSize(8, 8, 8);
		dim3 gridSize(hashMap->getNumAllocatedVoxelBlocks() / (blockSize.x * blockSize.y * blockSize.z));
		eraseMissingBlocksFromStdGpu<<<gridSize, blockSize>>>(map, hashMap->GetEntries(), voxels);
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
		float ratio = 1 / sum;
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
		ratio *= sum;
		printf("\tstdgpu: %f\n", sum);
		printf("\tratio: %.2f\n", ratio);
	}

	{
		dim3 blockSize(512);
		dim3 gridSize(307200 / 512);

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
		float ratio = 1 / sum;
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
		ratio *= sum;
		printf("\tstdgpu: %f\n", sum);
		printf("\tratio: %.2f\n", ratio);
	}

	{
		dim3 blockSize(512);
		dim3 gridSize(1000000 / 512);

		printf("Random read_uninterpolated of %i entries\n",
		       blockSize.x * blockSize.y * blockSize.z * gridSize.x * gridSize.y * gridSize.z);
		ITMTimer timer;
		float sum = 0;
		for (int i = 0; i < RUNS; i++)
		{
			timer.Tick();
			float* sum_device = createDeviceArray<float>(1, 0.0f);
			readUninterpolatedFromTSDFITM<<<gridSize, blockSize>>>(hashMap->GetEntries(), voxels, sum_device);
			ORcudaKernelCheck
			destroyDeviceArray(sum_device);
			sum += timer.Tock() / RUNS;
		}
		float ratio = 1 / sum;
		printf("\tITMLib: %f\n", sum);

		sum = 0;
		for (int i = 0; i < RUNS; i++)
		{
			timer.Tick();
			float* sum_device = createDeviceArray<float>(1, 0.0f);
			readUninterpolatedFromTSDFStdGpu<<<gridSize, blockSize>>>(map, sum_device);
			ORcudaKernelCheck
			destroyDeviceArray(sum_device);
			sum += timer.Tock() / RUNS;
		}
		ratio *= sum;
		printf("\tstdgpu: %f\n", sum);
		printf("\tratio: %.2f\n", ratio);
	}

	{
		dim3 blockSize(512);
		dim3 gridSize(1000000 / 512);

		printf("Random read_interpolated of %i entries\n",
		       blockSize.x * blockSize.y * blockSize.z * gridSize.x * gridSize.y * gridSize.z);
		ITMTimer timer;
		float sum = 0;
		for (int i = 0; i < RUNS; i++)
		{
			timer.Tick();
			float* sum_device = createDeviceArray<float>(1, 0.0f);
			readInterpolatedFromTSDFITM<<<gridSize, blockSize>>>(hashMap->GetEntries(), voxels, sum_device);
			ORcudaKernelCheck
			destroyDeviceArray(sum_device);
			sum += timer.Tock() / RUNS;
		}
		float ratio = 1 / sum;
		printf("\tITMLib: %f\n", sum);

		sum = 0;
		for (int i = 0; i < RUNS; i++)
		{
			timer.Tick();
			float* sum_device = createDeviceArray<float>(1, 0.0f);
			readInterpolatedFromTSDFStdGpu<<<gridSize, blockSize>>>(map, sum_device);
			ORcudaKernelCheck
			destroyDeviceArray(sum_device);
			sum += timer.Tock() / RUNS;
		}
		ratio *= sum;
		printf("\tstdgpu: %f\n", sum);
		printf("\tratio: %.2f\n", ratio);
	}

	{
		Vector2i imgSize(640, 480);
		dim3 blockSize(16, 12);
		dim3 gridSize((int) ceil((float) imgSize.x / (float) blockSize.x),
		              (int) ceil((float) imgSize.y / (float) blockSize.y));

		printf("raycasting of %i pixels\n",
		       blockSize.x * blockSize.y * blockSize.z * gridSize.x * gridSize.y * gridSize.z);
		ITMTimer timer;
		float sum = 0;
		for (int i = 0; i < RUNS; i++)
		{
			float* sum_device = createDeviceArray<float>(1, 0.0f);
			int* misses_device = createDeviceArray<int>(1, 0);
			timer.Tick();
			rayCastITM<<<gridSize, blockSize>>>(hashMap->GetEntries(), voxels, sum_device, misses_device);
			ORcudaKernelCheck;
			sum += timer.Tock() / RUNS;
			int misses; cudaMemcpy(&misses, misses_device, sizeof(int), cudaMemcpyDeviceToHost);
//			printf("%i, ", misses);
			destroyDeviceArray(sum_device);
			destroyDeviceArray(misses_device);
		}
		float ratio = 1 / sum;
		printf("\tITMLib: %f\n", sum);

		sum = 0;
		for (int i = 0; i < RUNS; i++)
		{
			float* sum_device = createDeviceArray<float>(1, 0.0f);
			int* misses_device = createDeviceArray<int>(1, 0);
			timer.Tick();
			rayCastStdGpu<<<gridSize, blockSize>>>(map, sum_device, misses_device);
			ORcudaKernelCheck
			sum += timer.Tock() / RUNS;
			int misses; cudaMemcpy(&misses, misses_device, sizeof(int), cudaMemcpyDeviceToHost);
//			printf("%i, ", misses);
			destroyDeviceArray(sum_device);
			destroyDeviceArray(misses_device);
		}
		ratio *= sum;
		printf("\tstdgpu: %f\n", sum);
		printf("\tratio: %.2f\n", ratio);
	}

	HashMapStdGpu::destroyDeviceObject(map);
	delete hashMap;
	delete voxels;

	return 0;
}