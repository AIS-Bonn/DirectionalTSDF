//
// Created by Malte Splietker on 28.12.22.
//

#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include "../Shared/ITMEvaluationEngine_Shared.h"
#include "ITMEvaluationEngine_CUDA.h"

#include <stdgpu/vector.cuh>
#include <Trackers/Shared/ITMICPTracker_Shared.h>
#include <ORUtils/CUDADefines.h>
#include <Utils/ITMCUDAUtils.h>

using namespace ITMLib;

ITMEvaluationEngine_CUDA::ITMEvaluationEngine_CUDA() = default;

struct ErrorAccumulator
{
	int noPoints;
	float error;
	float sqError;
	float icpError;
	float sqIcpError;
};

#define BlockSideLength 16

template<size_t blockSize>
__global__ void reduceAccumulator(ErrorAccumulator* accu, size_t N)
{
	size_t threadId = threadIdx.x + blockDim.x * threadIdx.y;
	size_t blockId = blockIdx.x + gridDim.x * blockIdx.y;

	ErrorAccumulator sum{0, 0.0f, 0.0f, 0.0f, 0.0f};
	for (size_t i = blockId * blockSize + threadId; i < N; i += blockDim.x * gridDim.x)
	{
		sum.noPoints += accu[i].noPoints;
		sum.error += accu[i].error;
		sum.sqError += accu[i].sqError;
		sum.icpError += accu[i].icpError;
		sum.sqIcpError += accu[i].sqIcpError;
	}

	parallelReduce<blockSize>(accu[blockId].noPoints, sum.noPoints, threadId);
	parallelReduce<blockSize>(accu[blockId].error, sum.error, threadId);
	parallelReduce<blockSize>(accu[blockId].sqError, sum.sqError, threadId);
	parallelReduce<blockSize>(accu[blockId].icpError, sum.icpError, threadId);
	parallelReduce<blockSize>(accu[blockId].sqIcpError, sum.sqIcpError, threadId);
}

__global__ void
computeICPError_device(
	const float* depth,
	const Vector4f* pointsRay,
	const Vector4f* normalsRay,
	const Matrix4f depthImageInvPose,
	const Matrix4f sceneRenderingPose,
	ITMIntrinsics intrinsics_d,
	ErrorAccumulator* accumulator
)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	size_t threadId = threadIdx.x + blockDim.x * threadIdx.y;
	size_t blockId = blockIdx.x + gridDim.x * blockIdx.y;

	__shared__ bool blockHasValidPoint;
	blockHasValidPoint = false;
	__syncthreads();

	if (x >= intrinsics_d.imgSize.width || y >= intrinsics_d.imgSize.height) return;

	float A[6];
	float error, icpError;
	float weight;
	bool isValidPoint = computePerPointGH_Depth_Ab<false, false>(
		A, icpError, weight, x, y,
		depthImageInvPose, sceneRenderingPose,
		depth,
		intrinsics_d.imgSize, intrinsics_d.projectionParamsSimple.all,
		intrinsics_d.imgSize, intrinsics_d.projectionParamsSimple.all,
		pointsRay, normalsRay, 100.0);

	isValidPoint &= computePerPointError<false, false>(
		error, x, y, depth,
		intrinsics_d.imgSize, intrinsics_d.projectionParamsSimple.all,
		intrinsics_d.imgSize, intrinsics_d.projectionParamsSimple.all,
		depthImageInvPose, sceneRenderingPose,
		pointsRay);

	if (isValidPoint)
		blockHasValidPoint = true;
	else
	{
		error = 0;
		icpError = 0;
	}

	__syncthreads();
	if (!blockHasValidPoint) return;

	parallelReduce<BlockSideLength * BlockSideLength>(accumulator[blockId].noPoints, (int) isValidPoint, threadId);
	parallelReduce<BlockSideLength * BlockSideLength>(accumulator[blockId].error, std::fabs(error), threadId);
	parallelReduce<BlockSideLength * BlockSideLength>(accumulator[blockId].sqError, error * error, threadId);
	parallelReduce<BlockSideLength * BlockSideLength>(accumulator[blockId].icpError, std::fabs(icpError), threadId);
	parallelReduce<BlockSideLength * BlockSideLength>(accumulator[blockId].sqIcpError, icpError * icpError, threadId);
}

__global__ void
computePhotometricError_device(
	const Vector4u* colorsImage,
	const Vector4u* colorsRender,
	const Vector4f* pointsRay,
	const Matrix4f depthImageInvPose,
	const Matrix4f sceneRenderingPose,
	ITMIntrinsics intrinsics_rgb,
	ErrorAccumulator* accumulator
)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	size_t threadId = threadIdx.x + blockDim.x * threadIdx.y;
	size_t blockId = blockIdx.x + gridDim.x * blockIdx.y;

	__shared__ bool blockHasValidPoint;
	blockHasValidPoint = false;
	__syncthreads();

	if (x >= intrinsics_rgb.imgSize.width || y >= intrinsics_rgb.imgSize.height) return;

	float error;

	int idx = PixelCoordsToIndex(x, y, intrinsics_rgb.imgSize);

	float intensityImage =
		(0.299f * colorsImage[idx].x + 0.587f * colorsImage[idx].y + 0.114f * colorsImage[idx].z) / 255.f;
	float intensityRender =
		(0.299f * colorsRender[idx].x + 0.587f * colorsRender[idx].y + 0.114f * colorsRender[idx].z) / 255.f;

	error = std::fabs(intensityImage - intensityRender);

	bool isValidPoint = intensityRender > 0 and pointsRay[idx].w >= 0;
	if (isValidPoint)
		blockHasValidPoint = true;
	else
		error = 0;

	parallelReduce<BlockSideLength * BlockSideLength>(accumulator[blockId].noPoints, (int) isValidPoint, threadId);
	parallelReduce<BlockSideLength * BlockSideLength>(accumulator[blockId].error, std::fabs(error), threadId);
	parallelReduce<BlockSideLength * BlockSideLength>(accumulator[blockId].sqError, error * error, threadId);
}

ITMRenderError ITMEvaluationEngine_CUDA::ComputeICPError(
	const ITMFloatImage& depth,
	const ORUtils::Image<ORUtils::Vector4<float>>& points,
	const ORUtils::Image<ORUtils::Vector4<float>>& normals,
	const Matrix4f& depthImageInvPose,
	const Matrix4f& sceneRenderingPose,
	ITMIntrinsics& intrinsics_d
) const
{
	const float* depthPtr = depth.GetData(MEMORYDEVICE_CUDA);
	const Vector4f* pointsRay = points.GetData(MEMORYDEVICE_CUDA);
	const Vector4f* normalsRay = normals.GetData(MEMORYDEVICE_CUDA);
	dim3 blockSize(BlockSideLength, BlockSideLength);
	dim3 gridSize((int) ceil((float) intrinsics_d.imgSize.width / (float) blockSize.x),
	              (int) ceil((float) intrinsics_d.imgSize.height / (float) blockSize.y));

	size_t numBlocks = gridSize.x * gridSize.y;
	ErrorAccumulator* accumulator_device;
	ORcudaSafeCall(cudaMalloc(&accumulator_device, numBlocks * sizeof(ErrorAccumulator)));
	ORcudaSafeCall(cudaMemset(accumulator_device, 0, numBlocks * sizeof(ErrorAccumulator)));

	computeICPError_device<<<gridSize, blockSize>>>(
		depthPtr, pointsRay, normalsRay, depthImageInvPose, sceneRenderingPose, intrinsics_d, accumulator_device
	);
	cudaDeviceSynchronize();

	reduceAccumulator<1024><<<1, 1024>>>(accumulator_device, numBlocks);

	ErrorAccumulator accumulator{};
	ORcudaSafeCall(cudaMemcpy(&accumulator, accumulator_device, sizeof(ErrorAccumulator), cudaMemcpyDeviceToHost));
	ORcudaSafeCall(cudaFree(accumulator_device));

	ITMRenderError result;
	result.MAE = accumulator.error / accumulator.noPoints;
	result.RMSE = std::sqrt(accumulator.sqError / accumulator.noPoints);

	result.icpMAE = accumulator.icpError / accumulator.noPoints;
	result.icpRMSE = std::sqrt(accumulator.sqIcpError / accumulator.noPoints);
	return result;
}

ITMRenderError
ITMEvaluationEngine_CUDA::ComputePhotometricError(const ITMUChar4Image& image, const ITMUChar4Image& render,
                                                  const ITMFloat4Image& points, const Matrix4f& depthImageInvPose,
                                                  const Matrix4f& sceneRenderingPose, ITMIntrinsics& intrinsics_rgb) const
{
	const Vector4u* imagePtr = image.GetData(MEMORYDEVICE_CUDA);
	const Vector4u* renderPtr = render.GetData(MEMORYDEVICE_CUDA);
	const Vector4f* pointsRay = points.GetData(MEMORYDEVICE_CUDA);

	dim3 blockSize(BlockSideLength, BlockSideLength);
	dim3 gridSize((int) ceil((float) intrinsics_rgb.imgSize.width / (float) blockSize.x),
	              (int) ceil((float) intrinsics_rgb.imgSize.height / (float) blockSize.y));

	size_t numBlocks = gridSize.x * gridSize.y;
	ErrorAccumulator* accumulator_device;
	ORcudaSafeCall(cudaMalloc(&accumulator_device, numBlocks * sizeof(ErrorAccumulator)));
	ORcudaSafeCall(cudaMemset(accumulator_device, 0, numBlocks * sizeof(ErrorAccumulator)));

	computePhotometricError_device<<<gridSize, blockSize>>>(
		imagePtr, renderPtr, pointsRay, depthImageInvPose, sceneRenderingPose, intrinsics_rgb, accumulator_device
	);
	cudaDeviceSynchronize();

	reduceAccumulator<1024><<<1, 1024>>>(accumulator_device, numBlocks);

	ErrorAccumulator accumulator{};
	ORcudaSafeCall(cudaMemcpy(&accumulator, accumulator_device, sizeof(ErrorAccumulator), cudaMemcpyDeviceToHost));
	ORcudaSafeCall(cudaFree(accumulator_device));

	ITMRenderError result;
	result.MAE = accumulator.error / accumulator.noPoints;
	result.RMSE = std::sqrt(accumulator.sqError / accumulator.noPoints);

	result.icpMAE = 0.0f;
	result.icpRMSE = 0.0f;
	return result;
}
