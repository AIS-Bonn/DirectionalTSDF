// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <ITMLib/Utils/ITMMath.h>
#include <stdgpu/functional.h>

template<class T>
inline __device__ void warpReduce(volatile T* sdata, int tid)
{
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

/**
 * Multi-layer parallel tree reduction.
 * @tparam T target type
 * @param target storage target of reduction
 * @param source value to add
 * @param locId_local local Id inside block
 */
template<size_t blockSize, typename T>
__device__
inline void parallelReduce(T& target, const T& source, const int threadId)
{
	__shared__ T sdata[blockSize];
	sdata[threadId] = source;
	__syncthreads();

	if (blockSize >= 1024)
	{
		if (threadId < 512)
		{ sdata[threadId] += sdata[threadId + 512]; }
		__syncthreads();
	}
	if (blockSize >= 512)
	{
		if (threadId < 256)
		{ sdata[threadId] += sdata[threadId + 256]; }
		__syncthreads();
	}
	if (blockSize >= 256)
	{
		if (threadId < 128)
		{ sdata[threadId] += sdata[threadId + 128]; }
		__syncthreads();
	}
	if (blockSize >= 128)
	{
		if (threadId < 64)
		{ sdata[threadId] += sdata[threadId + 64]; }
		__syncthreads();
	}
	if (threadId < 32) warpReduce(sdata, threadId);
	__syncthreads();
	if (threadId == 0) target = sdata[0];
}

/**
 * Three layer parallel tree reduction. Uses atomicAdd for summing results from block reduction (results may not be reproducible)
 * @tparam T target type
 * @param target summation target of reduction (atomicAdd)
 * @param source value to add
 * @param threadId local Id inside block
 */
template<size_t blockSize, typename T>
__device__
inline void parallelReduceAtomic(T& target, const T& source, const int threadId)
{
	T result;
	parallelReduce<blockSize, T>(result, source, threadId);
	if (threadId == 0) atomicAdd(&target, result);
}

template<size_t blockSize, typename T>
__device__
inline void parallelReduceArray3(T* target, const T* source, const int threadId)
{

	__shared__ T sdata1[blockSize];
	__shared__ T sdata2[blockSize];
	__shared__ T sdata3[blockSize];

	sdata1[threadId] = source[0];
	sdata2[threadId] = source[1];
	sdata3[threadId] = source[2];
	__syncthreads();

	if (blockSize >= 1024)
	{
		if (threadId < 512)
		{
			sdata1[threadId] += sdata1[threadId + 512];
			sdata2[threadId] += sdata2[threadId + 512];
			sdata3[threadId] += sdata3[threadId + 512];
		}
		__syncthreads();
	}
	if (blockSize >= 512)
	{
		if (threadId < 256)
		{
			sdata1[threadId] += sdata1[threadId + 256];
			sdata2[threadId] += sdata2[threadId + 256];
			sdata3[threadId] += sdata3[threadId + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256)
	{
		if (threadId < 128)
		{
			sdata1[threadId] += sdata1[threadId + 128];
			sdata2[threadId] += sdata2[threadId + 128];
			sdata3[threadId] += sdata3[threadId + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128)
	{
		if (threadId < 64)
		{
			sdata1[threadId] += sdata1[threadId + 64];
			sdata2[threadId] += sdata2[threadId + 64];
			sdata3[threadId] += sdata3[threadId + 64];
		}
		__syncthreads();
	}

	if (threadId < 32)
	{
		warpReduce(sdata1, threadId);
		warpReduce(sdata2, threadId);
		warpReduce(sdata3, threadId);
	}
	__syncthreads();

	if (threadId == 0)
	{
		target[0] = sdata1[0];
		target[1] = sdata2[0];
		target[2] = sdata3[0];
	}
}

template<size_t blockSize, typename T>
__device__
inline void parallelReduceArray4(T* target, const T* source, const int threadId)
{

	__shared__ T sdata1[blockSize];
	__shared__ T sdata2[blockSize];
	__shared__ T sdata3[blockSize];
	__shared__ T sdata4[blockSize];

	sdata1[threadId] = source[0];
	sdata2[threadId] = source[1];
	sdata3[threadId] = source[2];
	sdata4[threadId] = source[3];
	__syncthreads();

	if (blockSize >= 1024)
	{
		if (threadId < 512)
		{
			sdata1[threadId] += sdata1[threadId + 512];
			sdata2[threadId] += sdata2[threadId + 512];
			sdata3[threadId] += sdata3[threadId + 512];
			sdata4[threadId] += sdata4[threadId + 512];
		}
		__syncthreads();
	}
	if (blockSize >= 512)
	{
		if (threadId < 256)
		{
			sdata1[threadId] += sdata1[threadId + 256];
			sdata2[threadId] += sdata2[threadId + 256];
			sdata3[threadId] += sdata3[threadId + 256];
			sdata4[threadId] += sdata4[threadId + 256];
		}
		__syncthreads();
	}
	if (blockSize >= 256)
	{
		if (threadId < 128)
		{
			sdata1[threadId] += sdata1[threadId + 128];
			sdata2[threadId] += sdata2[threadId + 128];
			sdata3[threadId] += sdata3[threadId + 128];
			sdata4[threadId] += sdata4[threadId + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128)
	{
		if (threadId < 64)
		{
			sdata1[threadId] += sdata1[threadId + 64];
			sdata2[threadId] += sdata2[threadId + 64];
			sdata3[threadId] += sdata3[threadId + 64];
			sdata4[threadId] += sdata4[threadId + 64];
		}
		__syncthreads();
	}

	if (threadId < 32)
	{
		warpReduce(sdata1, threadId);
		warpReduce(sdata2, threadId);
		warpReduce(sdata3, threadId);
		warpReduce(sdata4, threadId);
	}
	__syncthreads();

	if (threadId == 0)
	{
		target[0] = sdata1[0];
		target[1] = sdata2[0];
		target[2] = sdata3[0];
		target[3] = sdata4[0];
	}
}

/**
 * Three layer parallel tree reduction for arrays of 3 values. Uses atomicAdd for summing results from block reduction (results may not be reproducible)
 * @tparam T array type
 * @param target storage target of reduction
 * @param source array to add
 * @param locId_local local Id inside block
 */
template<size_t blockSize, typename T>
__device__
inline void parallelReduceArray3Atomic(T* target, const T* source, const int threadId)
{
	T result[3];
	parallelReduceArray3<blockSize, T>(result, source, threadId);
	if (threadId == 0)
	{
		atomicAdd(&target[0], result[0]);
		atomicAdd(&target[1], result[1]);
		atomicAdd(&target[2], result[2]);
	}
}

/**
 * Three layer parallel tree reduction for arrays of 4 values. Uses atomicAdd for summing results from block reduction (results may not be reproducible)
 * @tparam T array type
 * @param target storage target of reduction
 * @param source array to add
 * @param locId_local local Id inside block
 */
template<size_t blockSize, typename T>
__device__
inline void parallelReduceArray4Atomic(T* target, const T* source, const int threadId)
{
	T result[4];
	parallelReduceArray4<blockSize, T>(result, source, threadId);
	if (threadId == 0)
	{
		atomicAdd(&target[0], result[0]);
		atomicAdd(&target[1], result[1]);
		atomicAdd(&target[2], result[2]);
		atomicAdd(&target[3], result[3]);
	}
}

/**
 * Three layer parallel tree reduction for Vector3 types
 * @tparam T vector type
 * @param target storage target of reduction
 * @param source value to add
 * @param threadId local Id inside block
 */
template<size_t blockSize, typename T>
__device__
inline void parallelReduceVector3(ORUtils::Vector3<T>& target, const ORUtils::Vector3<T>& source, const int threadId)
{
	parallelReduceArray3Atomic<blockSize, T>(target.v, source.v, threadId);
}

/**
 * Three layer parallel tree reduction for Vector4 types
 * @tparam T vector type
 * @param target storage target of reduction
 * @param source value to add
 * @param threadId local Id inside block
 */
template<size_t blockSize, typename T>
__device__
inline void parallelReduceVector4(ORUtils::Vector4<T>& target, const ORUtils::Vector4<T>& source, const int threadId)
{
	parallelReduceArray4Atomic<blockSize, T>(target.v, source.v, threadId);
}

template<size_t blockSize, typename T>
__global__ void reduceBlock_device(T* out, const T* in, size_t N)
{
	size_t threadId = threadIdx.x + blockDim.x * threadIdx.y;
	size_t blockId = blockIdx.x + gridDim.x * blockIdx.y;
	size_t i = blockId * blockSize + threadId;

	T data = 0;
	if (i < N) data = in[i];

	parallelReduce<blockSize>(out[blockId], data, threadId);
}

/**
 * Reduce an array of type T by parallel reduction with pyramid (no atomic operation)
 * @tparam blockSize size of individual blocks
 * @tparam T type of data
 * @param data input data
 * @param N number elements in data
 * @return
 */
template<size_t blockSize, typename T>
T GPUReduction(T* data, size_t N)
{
	size_t n = N;
	size_t blocksPerGrid = std::ceil((float) n / blockSize);

	T* kernelOut;
	ORcudaSafeCall(cudaMalloc(&kernelOut, sizeof(T) * blocksPerGrid));
	T* kernelIn = data;

	// pyramid reduction with blocks into kernelOut, until last remaining reduction fits into single block
	do
	{
		blocksPerGrid = std::ceil((float) n / blockSize);
		reduceBlock_device<blockSize><<<blocksPerGrid, blockSize>>>(kernelOut, kernelIn, n);
		kernelIn = kernelOut;
		n = blocksPerGrid;
	} while (n > blockSize);

	if (n > 1)
		reduceBlock_device<blockSize><<<1, blockSize>>>(kernelOut, kernelOut, n);

	cudaDeviceSynchronize();

	T sum;
	cudaMemcpy(&sum, &kernelOut[0], sizeof(T), cudaMemcpyDeviceToHost);
	ORcudaSafeCall(cudaFree(kernelOut));
	return sum;
}

template<typename T>
__device__ int computePrefixSum_device(uint element, T* sum, int localSize, int localId)
{
	// TODO: should be localSize...
	__shared__ uint prefixBuffer[16 * 16];
	__shared__ uint groupOffset;

	prefixBuffer[localId] = element;
	__syncthreads();

	int s1, s2;

	for (s1 = 1, s2 = 1; s1 < localSize; s1 <<= 1)
	{
		s2 |= s1;
		if ((localId & s2) == s2) prefixBuffer[localId] += prefixBuffer[localId - s1];
		__syncthreads();
	}

	for (s1 >>= 2, s2 >>= 1; s1 >= 1; s1 >>= 1, s2 >>= 1)
	{
		if (localId != localSize - 1 && (localId & s2) == s2) prefixBuffer[localId + s1] += prefixBuffer[localId];
		__syncthreads();
	}

	if (localId == 0 && prefixBuffer[localSize - 1] > 0) groupOffset = atomicAdd(sum, prefixBuffer[localSize - 1]);
	__syncthreads();

	int offset;// = groupOffset + prefixBuffer[localId] - 1;
	if (localId == 0)
	{
		if (prefixBuffer[localId] == 0) offset = -1;
		else offset = groupOffset;
	} else
	{
		if (prefixBuffer[localId] == prefixBuffer[localId - 1]) offset = -1;
		else offset = groupOffset + prefixBuffer[localId - 1];
	}

	return offset;
}

//__device__ static inline void atomicMin(float* address, float val)
//{
//	int* address_as_i = (int*)address;
//	int old = *address_as_i, assumed;
//	do {
//		assumed = old;
//		old = ::atomicCAS(address_as_i, assumed,
//			__float_as_int(::fminf(val, __int_as_float(assumed))));
//	} while (assumed != old);
//}

//__device__ static inline void atomicMax(float* address, float val)
//{
//	int* address_as_i = (int*)address;
//	int old = *address_as_i, assumed;
//	do {
//		assumed = old;
//		old = ::atomicCAS(address_as_i, assumed,
//			__float_as_int(::fmaxf(val, __int_as_float(assumed))));
//	} while (assumed != old);
//}

template<typename T>
__global__ void memsetKernel_device(T* devPtr, const T val, size_t nwords)
{
	size_t offset = threadIdx.x + blockDim.x * blockIdx.x;
	if (offset >= nwords) return;
	devPtr[offset] = val;
}

template<typename T>
__global__ void memsetKernelLarge_device(T* devPtr, const T val, size_t nwords)
{
	size_t offset = threadIdx.x + blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x);
	if (offset >= nwords) return;
	devPtr[offset] = val;
}

template<typename T>
inline void memsetKernel(T* devPtr, const T val, size_t nwords)
{
	dim3 blockSize(256);
	dim3 gridSize((int) ceil((float) nwords / (float) blockSize.x));
	if (gridSize.x <= 65535)
	{
		memsetKernel_device<T> <<<gridSize, blockSize>>>(devPtr, val, nwords);
		ORcudaKernelCheck;
	} else
	{
		gridSize.x = (int) ceil(sqrt((float) gridSize.x));
		gridSize.y = (int) ceil((float) nwords / (float) (blockSize.x * gridSize.x));
		memsetKernelLarge_device<T> <<<gridSize, blockSize>>>(devPtr, val, nwords);
		ORcudaKernelCheck;
	}
}

template<typename T>
__global__ void fillArrayKernel_device(T* devPtr, size_t nwords)
{
	size_t offset = threadIdx.x + blockDim.x * blockIdx.x;
	if (offset >= nwords) return;
	devPtr[offset] = offset;
}

template<typename T>
inline void fillArrayKernel(T* devPtr, size_t nwords)
{
	dim3 blockSize(256);
	dim3 gridSize((int) ceil((float) nwords / (float) blockSize.x));
	fillArrayKernel_device<T> <<<gridSize, blockSize>>>(devPtr, nwords);
	ORcudaKernelCheck;
}

