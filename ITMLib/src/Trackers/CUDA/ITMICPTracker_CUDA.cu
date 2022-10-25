// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMICPTracker_CUDA.h"
#include "../Shared/ITMICPTracker_Shared.h"
#include <Trackers/Shared/ITMTracker_Shared.h>
#include <Utils/ITMCUDAUtils.h>
#include <ORUtils/CUDADefines.h>
#include <ORUtils/EigenConversion.h>
#include <ORUtils/FileUtils.h>

using namespace ITMLib;

struct ITMICPTracker_CUDA::AccuCell
{
	int numPoints;
	float f;
	float g[6];
	float h[6 + 5 + 4 + 3 + 2 + 1];

	_CPU_AND_GPU_CODE_
	void SetZero()
	{
		numPoints = 0;
		f = 0;
#pragma unroll
		for (float& i : g)
			i = 0;
#pragma unroll
		for (float& i : h)
			i = 0;
	}
};

struct AccumulatorSim3
{
	static const int sizeH = 7 + 6 + 5 + 4 + 3 + 2 + 1; // size of lower triangle matrix
	static const int sizeg = 7;

	int numPoints;
	float H[7 + 6 + 5 + 4 + 3 + 2 + 1]; // diagonal halfmatrix of Hessian
	float g[7]; // residual Vector
	float f; // error

	_CPU_AND_GPU_CODE_
	void SetZero()
	{
		numPoints = 0;
		f = 0;
#pragma unroll
		for (float& i : H)
			i = 0;
#pragma unroll
		for (float& i : g)
			i = 0;
	}
};

struct KernelParameters
{
	ITMICPTracker_CUDA::AccuCell* accu;
	float* depth;
	Matrix4f deltaT;
	Vector4f* pointsMap;
	Vector4f* normalsMap;
	Vector4f sceneIntrinsics;
	Vector2i sceneImageSize;
	Matrix4f T_scene_CW;
	Vector4f viewIntrinsics;
	Vector2i viewImageSize;
	float distThresh;
};

struct RGBKernelParameters
{
	ITMICPTracker_CUDA::AccuCell* accu;
	Vector4f* points_current;
	float* intensity_current;
	float* intensity_reference;
	Vector2f* gradient_reference;
	Matrix4f deltaT;
	Matrix4f T_ref_CW;
	Matrix4f T_scene_WC;
	Vector4f intrinsics_depth;
	Vector4f intrinsics_rgb;
	Vector2i imgSize_depth;
	Vector2i imgSize_rgb;
	float viewFrustum_max;
	float intensityThresh;
	float minGradient;
};

__global__ void
Sim3DerivativeDepth_device(AccumulatorSim3* accumulator,
                           const float* depth, const Vector4f* pointsMap, const Vector4f* normalsMap,
                           Vector4f sceneIntrinsics, Vector2i sceneImageSize,
                           Vector4f viewIntrinsics, Vector2i viewImageSize,
                           Matrix4f approxInvPose, Matrix4f scenePose,
                           float distThresh);

__global__ void
Sim3DerivativeRGB_device(AccumulatorSim3* accumulator,
                         Vector4f* points_current,
                         float* intensity_current, float* intensity_reference, Vector2f* gradient_reference,
                         Vector4f intrinsics_depth, Vector4f intrinsics_rgb,
                         Vector2i imgSize_depth, Vector2i imgSize_rgb,
                         Matrix4f deltaT, Matrix4f T_ref_CW, Matrix4f T_scene_WC,
                         float viewFrustum_max,
                         float intensityThresh,
                         float minGradient);

template<bool shortIteration, bool rotationOnly>
__global__ void
depthTrackerOneLevel_g_rt_device(KernelParameters para);

template<bool shortIteration, bool rotationOnly>
__global__ void RGBTrackerOneLevel_g_rt_device(RGBKernelParameters para);

// host methods

#define BlockSideLength 16
dim3 computationBlockSize(BlockSideLength, BlockSideLength);

ITMICPTracker_CUDA::ITMICPTracker_CUDA(Vector2i imgSize_d, Vector2i imgSize_rgb, const Parameters& parameters,
                                       const ITMLowLevelEngine* lowLevelEngine)
	: ITMICPTracker(imgSize_d, imgSize_rgb, parameters, lowLevelEngine, MEMORYDEVICE_CUDA)
{
	dim3 gridSize((int) std::ceil((float) imgSize_d.x / (float) computationBlockSize.x),
	              (int) std::ceil((float) imgSize_d.y / (float) computationBlockSize.y));
	size_t numBlocks = gridSize.x * gridSize.y;
	ORcudaSafeCall(cudaMalloc(&accumulator_device, numBlocks * sizeof(AccuCell)));
}

ITMICPTracker_CUDA::~ITMICPTracker_CUDA(void)
{
	ORcudaSafeCall(cudaFree(accumulator_device));
}

/** Perform final reduction of per-block summation values stored in accumulator array
 * */
template<size_t blockSize>
__global__ void reduceAccumulatorSim3(AccumulatorSim3* accu, size_t N)
{
	size_t threadId = threadIdx.x + blockDim.x * threadIdx.y;
	size_t blockId = blockIdx.x + gridDim.x * blockIdx.y;

	AccumulatorSim3 sum{};
	sum.SetZero();

	for (size_t i = blockId * blockSize + threadId; i < N; i += blockDim.x * gridDim.x)
	{
		sum.numPoints += accu[i].numPoints;
		sum.f += accu[i].f;
		for (int j = 0; j < 7; j++)
			sum.g[j] += accu[i].g[j];
		for (int j = 0; j < AccumulatorSim3::sizeH; j++)
			sum.H[j] += accu[i].H[j];
	}

	parallelReduce<blockSize>(accu[blockId].numPoints, sum.numPoints, threadId);
	parallelReduce<blockSize>(accu[blockId].f, sum.f, threadId);
	parallelReduceArray4<blockSize>(accu[blockId].g, sum.g, threadId);
	parallelReduceArray3<blockSize>(accu[blockId].g + 4, sum.g + 4, threadId);
	for (unsigned char offset = 0; offset < AccumulatorSim3::sizeH; offset += 4)
		parallelReduceArray4<blockSize>(accu[blockId].H + offset, sum.H + offset, threadId);
}

size_t
ITMICPTracker_CUDA::ComputeGandHSim3_Depth(float& f, Eigen::Matrix<EigenT, 7, 7>& H, Eigen::Matrix<EigenT, 7, 1>& g,
                                           const Matrix4f& deltaT)
{
	Vector4f* pointsMap = sceneHierarchy->GetLevel(0)->pointsMap->GetData(MEMORYDEVICE_CUDA);
	Vector4f* normalsMap = sceneHierarchy->GetLevel(0)->normalsMap->GetData(MEMORYDEVICE_CUDA);
	Vector4f sceneIntrinsics = sceneHierarchy->GetLevel(0)->intrinsics;
	Vector2i sceneImageSize = sceneHierarchy->GetLevel(0)->pointsMap->noDims;

	float* depth = viewHierarchy_depth->GetLevel(levelId)->data->GetData(MEMORYDEVICE_CUDA);
	Vector4f viewIntrinsics = viewHierarchy_depth->GetLevel(levelId)->intrinsics;
	Vector2i viewImageSize = viewHierarchy_depth->GetLevel(levelId)->data->noDims;

	dim3 gridSize((int) std::ceil((float) viewImageSize.x / (float) computationBlockSize.x),
	              (int) std::ceil((float) viewImageSize.y / (float) computationBlockSize.y));

	size_t numBlocks = gridSize.x * gridSize.y;

	AccumulatorSim3* accumulator_device;
	ORcudaSafeCall(cudaMalloc(&accumulator_device, numBlocks * sizeof(AccumulatorSim3)));
	ORcudaSafeCall(cudaMemset(accumulator_device, 0, numBlocks * sizeof(AccumulatorSim3)));

	Sim3DerivativeDepth_device << < gridSize, computationBlockSize >> >(accumulator_device, depth, pointsMap, normalsMap,
		sceneIntrinsics, sceneImageSize, viewIntrinsics, viewImageSize,
		deltaT, renderedScenePose, distThresh[levelId]);
	reduceAccumulatorSim3<1024><<<1, 1024>>>(accumulator_device, numBlocks);
	ORcudaKernelCheck;

	AccumulatorSim3 accumulator{};
	ORcudaSafeCall(cudaMemcpy(&accumulator, accumulator_device, sizeof(AccumulatorSim3), cudaMemcpyDeviceToHost));
	ORcudaSafeCall(cudaFree(accumulator_device));

	if (accumulator.numPoints == 0)
		return 0;

	float invNoPoints = 1.0f / accumulator.numPoints;

	for (int r = 0, counter = 0; r < 7; r++)
		for (int c = 0; c <= r; c++, counter++)
			H(r, c) = accumulator.H[counter];
	for (int r = 0; r < 7; ++r)
		for (int c = r + 1; c < 7; c++)
			H(r, c) = H(c, r);

	for (int c = 0; c < 7; c++)
		g[c] = accumulator.g[c];

	f = accumulator.f * invNoPoints;

	return accumulator.numPoints;
}

size_t
ITMICPTracker_CUDA::ComputeGandHSim3_RGB(float& f, Eigen::Matrix<EigenT, 7, 7>& H, Eigen::Matrix<EigenT, 7, 1>& g,
                                         const Matrix4f& deltaT)
{
	Vector2i viewImageSize = viewHierarchy_depth->GetLevel(levelId)->data->noDims;

	dim3 gridSize((int) std::ceil((float) viewImageSize.x / (float) computationBlockSize.x),
	              (int) std::ceil((float) viewImageSize.y / (float) computationBlockSize.y));

	size_t numBlocks = gridSize.x * gridSize.y;

	AccumulatorSim3* accumulator_device;
	ORcudaSafeCall(cudaMalloc(&accumulator_device, numBlocks * sizeof(AccumulatorSim3)));
	ORcudaSafeCall(cudaMemset(accumulator_device, 0, numBlocks * sizeof(AccumulatorSim3)));


	Sim3DerivativeRGB_device << < gridSize, computationBlockSize >> >(
		accumulator_device,
			reprojectedPointsHierarchy->GetLevel(levelId)->data->GetData(MEMORYDEVICE_CUDA),
			projectedIntensityHierarchy->GetLevel(levelId)->data->GetData(MEMORYDEVICE_CUDA),
			viewHierarchy_intensity->GetLevel(levelId)->intensity_prev->GetData(MEMORYDEVICE_CUDA),
			viewHierarchy_intensity->GetLevel(levelId)->gradients->GetData(MEMORYDEVICE_CUDA),
			viewHierarchy_depth->GetLevel(levelId)->intrinsics,
			viewHierarchy_intensity->GetLevel(levelId)->intrinsics,
			viewHierarchy_depth->GetLevel(levelId)->data->noDims,
			viewHierarchy_intensity->GetLevel(levelId)->intensity_prev->noDims,
			deltaT,
			depthToRGBTransform * intensityReferencePose,
			renderedScenePose.inv(),
			6, // FIXME: use parameter
			colourThresh[levelId],
			parameters.minColourGradient
	);
	reduceAccumulatorSim3<1024><<<1, 1024>>>(accumulator_device, numBlocks);
	ORcudaKernelCheck;

	AccumulatorSim3 accumulator{};
	ORcudaSafeCall(cudaMemcpy(&accumulator, accumulator_device, sizeof(AccumulatorSim3), cudaMemcpyDeviceToHost));
	ORcudaSafeCall(cudaFree(accumulator_device));

	if (accumulator.numPoints == 0)
		return 0;

	float invNoPoints = 1.0f / accumulator.numPoints;

	for (int r = 0, counter = 0; r < 7; r++)
		for (int c = 0; c <= r; c++, counter++)
			H(r, c) = accumulator.H[counter];
	for (int r = 0; r < 7; ++r)
		for (int c = r + 1; c < 7; c++)
			H(r, c) = H(c, r);

	for (int c = 0; c < 7; c++)
		g[c] = accumulator.g[c];

	f = accumulator.f * invNoPoints;

	return accumulator.numPoints;
}

/** Perform final reduction of per-block summation values stored in accumulator array
 * */
template<bool shortIteration, size_t blockSize>
__global__ void reduceAccumulator(ITMICPTracker_CUDA::AccuCell* accu, size_t N)
{
	size_t threadId = threadIdx.x + blockDim.x * threadIdx.y;
	size_t blockId = blockIdx.x + gridDim.x * blockIdx.y;

	const int noPara = shortIteration ? 3 : 6;
	const int noParaSQ = shortIteration ? 3 + 2 + 1 : 6 + 5 + 4 + 3 + 2 + 1;

	ITMICPTracker_CUDA::AccuCell sum{};
	sum.SetZero();
	for (size_t i = blockId * blockSize + threadId; i < N; i += blockDim.x * gridDim.x)
	{
		sum.numPoints += accu[i].numPoints;
		sum.f += accu[i].f;
#pragma unroll
		for (int j = 0; j < noPara; j++)
			sum.g[j] += accu[i].g[j];
#pragma unroll
		for (int j = 0; j < noParaSQ; j++)
			sum.h[j] += accu[i].h[j];
	}

	parallelReduce<blockSize>(accu[blockId].numPoints, sum.numPoints, threadId);
	parallelReduce<blockSize>(accu[blockId].f, sum.f, threadId);
	for (unsigned char paraId = 0; paraId < noPara; paraId += 3)
		parallelReduceArray3<blockSize>(accu[blockId].g + paraId, sum.g + paraId, threadId);
	for (unsigned char paraId = 0; paraId < noParaSQ; paraId += 3)
		parallelReduceArray3<blockSize>(accu[blockId].h + paraId, sum.h + paraId, threadId);
}

int ITMICPTracker_CUDA::ComputeGandH_Depth(float& f, float* nabla, float* hessian, const Matrix4f& deltaT)
{
	Vector4f* pointsMap = sceneHierarchy->GetLevel(0)->pointsMap->GetData(MEMORYDEVICE_CUDA);
	Vector4f* normalsMap = sceneHierarchy->GetLevel(0)->normalsMap->GetData(MEMORYDEVICE_CUDA);
	Vector4f sceneIntrinsics = sceneHierarchy->GetLevel(0)->intrinsics;
	Vector2i sceneImageSize = sceneHierarchy->GetLevel(0)->pointsMap->noDims;

	float* depth = viewHierarchy_depth->GetLevel(levelId)->data->GetData(MEMORYDEVICE_CUDA);
	Vector4f viewIntrinsics = viewHierarchy_depth->GetLevel(levelId)->intrinsics;
	Vector2i viewImageSize = viewHierarchy_depth->GetLevel(levelId)->data->noDims;

	if (iterationType == TRACKER_ITERATION_NONE) return 0;

	bool shortIteration =
		(iterationType == TRACKER_ITERATION_ROTATION) || (iterationType == TRACKER_ITERATION_TRANSLATION);

	int noPara = shortIteration ? 3 : 6;

	dim3 gridSize((int) std::ceil((float) viewImageSize.x / (float) computationBlockSize.x),
	              (int) std::ceil((float) viewImageSize.y / (float) computationBlockSize.y));

	size_t numBlocks = gridSize.x * gridSize.y;
	ORcudaSafeCall(cudaMemset(accumulator_device, 0, numBlocks * sizeof(AccuCell)));
	struct KernelParameters args;
	args.accu = accumulator_device;
	args.depth = depth;
	args.deltaT = deltaT;
	args.T_scene_CW = renderedScenePose;
	args.pointsMap = pointsMap;
	args.normalsMap = normalsMap;
	args.sceneIntrinsics = sceneIntrinsics;
	args.sceneImageSize = sceneImageSize;
	args.viewIntrinsics = viewIntrinsics;
	args.viewImageSize = viewImageSize;
	args.distThresh = distThresh[levelId];

	switch (iterationType)
	{
		case TRACKER_ITERATION_ROTATION:
			depthTrackerOneLevel_g_rt_device<true, true> << < gridSize, computationBlockSize >> >(args);
			ORcudaKernelCheck;
			reduceAccumulator<true, 1024><<<1, 1024>>>(accumulator_device, numBlocks);
			ORcudaKernelCheck;
			break;
		case TRACKER_ITERATION_TRANSLATION:
			depthTrackerOneLevel_g_rt_device<true, false> << < gridSize, computationBlockSize >> >(args);
			ORcudaKernelCheck;
			reduceAccumulator<true, 1024><<<1, 1024>>>(accumulator_device, numBlocks);
			ORcudaKernelCheck;
			break;
		case TRACKER_ITERATION_BOTH:
			depthTrackerOneLevel_g_rt_device<false, false> << < gridSize, computationBlockSize >> >(args);
			ORcudaKernelCheck;
			reduceAccumulator<false, 1024><<<1, 1024>>>(accumulator_device, numBlocks);
			ORcudaKernelCheck;
			break;
		default:
			break;
	}
	AccuCell accumulator{};
	ORcudaSafeCall(cudaMemcpy(&accumulator, accumulator_device, sizeof(AccuCell), cudaMemcpyDeviceToHost));

	if (accumulator.numPoints == 0)
		return 0;

	float invNoPoints = 1.0f / accumulator.numPoints;

	for (int r = 0, counter = 0; r < noPara; r++)
		for (int c = 0; c <= r; c++, counter++)
			hessian[r + c * 6] = accumulator.h[counter] * invNoPoints;
	for (int r = 0; r < noPara; ++r)
		for (int c = r + 1; c < noPara; c++)
			hessian[r + c * 6] = hessian[c + r * 6];

	for (int c = 0; c < noPara; c++)
		nabla[c] = accumulator.g[c] * invNoPoints;

	f = accumulator.f * invNoPoints;

	return accumulator.numPoints;
}

__global__ void
RGBTrackerErrorImage_device(Vector4u* outRendering,
                            RGBKernelParameters para,
                            float maxError);

int ITMICPTracker_CUDA::ComputeGandH_RGB(float& f, float* nabla, float* hessian, const Matrix4f& deltaT)
{
	Vector2i imageSize_depth = viewHierarchy_depth->GetLevel(levelId)->data->noDims;
	Vector2i imageSize_rgb = viewHierarchy_depth->GetLevel(levelId)->data->noDims;

	if (iterationType == TRACKER_ITERATION_NONE) return 0;

	bool shortIteration = iterationType == TRACKER_ITERATION_ROTATION
	                      || iterationType == TRACKER_ITERATION_TRANSLATION;
	int noPara = shortIteration ? 3 : 6;

	dim3 gridSize((int) std::ceil((float) imageSize_depth.x / (float) computationBlockSize.x),
	              (int) std::ceil((float) imageSize_depth.y / (float) computationBlockSize.y));
	size_t numBlocks = gridSize.x * gridSize.y;

	ORcudaSafeCall(cudaMemset(accumulator_device, 0, numBlocks * sizeof(AccuCell)));

	struct RGBKernelParameters args;
	args.accu = accumulator_device;
	args.points_current = reprojectedPointsHierarchy->GetLevel(levelId)->data->GetData(MEMORYDEVICE_CUDA);
	args.intensity_current = projectedIntensityHierarchy->GetLevel(levelId)->data->GetData(MEMORYDEVICE_CUDA);
	args.intensity_reference = viewHierarchy_intensity->GetLevel(levelId)->intensity_prev->GetData(MEMORYDEVICE_CUDA);
	args.gradient_reference = viewHierarchy_intensity->GetLevel(levelId)->gradients->GetData(MEMORYDEVICE_CUDA);
	args.deltaT = deltaT;
	args.intrinsics_depth = viewHierarchy_depth->GetLevel(levelId)->intrinsics;
	args.imgSize_depth = viewHierarchy_depth->GetLevel(levelId)->data->noDims;
	args.intrinsics_rgb = viewHierarchy_intensity->GetLevel(levelId)->intrinsics;
	args.imgSize_rgb = viewHierarchy_intensity->GetLevel(levelId)->intensity_prev->noDims;
	args.T_ref_CW = depthToRGBTransform * intensityReferencePose;
	args.T_scene_WC = renderedScenePose.inv();
	args.viewFrustum_max = 6; // FIXME: use parameter
	args.intensityThresh = colourThresh[levelId];
	args.minGradient = parameters.minColourGradient;

	switch (iterationType)
	{
		case TRACKER_ITERATION_ROTATION:
			RGBTrackerOneLevel_g_rt_device<true, true> << < gridSize, computationBlockSize >> >(args);
			ORcudaKernelCheck;
			reduceAccumulator<true, 1024><<<1, 1024>>>(accumulator_device, numBlocks);
			ORcudaKernelCheck;
			break;
		case TRACKER_ITERATION_TRANSLATION:
			RGBTrackerOneLevel_g_rt_device<true, false> << < gridSize, computationBlockSize >> >(args);
			ORcudaKernelCheck;
			reduceAccumulator<true, 1024><<<1, 1024>>>(accumulator_device, numBlocks);
			ORcudaKernelCheck;
			break;
		case TRACKER_ITERATION_BOTH:
			RGBTrackerOneLevel_g_rt_device<false, false> << < gridSize, computationBlockSize >> >(args);
			ORcudaKernelCheck;
			reduceAccumulator<false, 1024><<<1, 1024>>>(accumulator_device, numBlocks);
			ORcudaKernelCheck;
			break;
		default:
			break;
	}
	ITMICPTracker_CUDA::AccuCell accu_host{};
	ORcudaSafeCall(cudaMemcpy(&accu_host, accumulator_device, sizeof(AccuCell), cudaMemcpyDeviceToHost));

	if (accu_host.numPoints == 0)
		return 0;

	float invNoPoints = 1.0f / accu_host.numPoints;

	for (int r = 0, counter = 0; r < noPara; r++)
		for (int c = 0; c <= r; c++, counter++)
			hessian[r + c * 6] = accu_host.h[counter] * invNoPoints;
	for (int r = 0; r < noPara; ++r)
		for (int c = r + 1; c < noPara; c++)
			hessian[r + c * 6] = hessian[c + r * 6];

	for (int c = 0; c < noPara; c++)
		nabla[c] = accu_host.g[c] * invNoPoints;

	f = accu_host.f * invNoPoints;

	return accu_host.numPoints;
}

void ITMICPTracker_CUDA::RenderRGBError(ITMUChar4Image* image_out, const Matrix4f& deltaT)
{
	Vector2i imageSize_depth = viewHierarchy_depth->GetLevel(levelId)->data->noDims;
	Vector2i imageSize_rgb = viewHierarchy_depth->GetLevel(levelId)->data->noDims;

	if (iterationType == TRACKER_ITERATION_NONE) return;

	bool shortIteration = iterationType == TRACKER_ITERATION_ROTATION
	                      || iterationType == TRACKER_ITERATION_TRANSLATION;
	int noPara = shortIteration ? 3 : 6;

	dim3 gridSize((int) std::ceil((float) imageSize_depth.x / (float) computationBlockSize.x),
	              (int) std::ceil((float) imageSize_depth.y / (float) computationBlockSize.y));
	size_t numBlocks = gridSize.x * gridSize.y;

	ORcudaSafeCall(cudaMemset(accumulator_device, 0, numBlocks * sizeof(AccuCell)));

	struct RGBKernelParameters args;
	args.accu = accumulator_device;
	args.points_current = reprojectedPointsHierarchy->GetLevel(levelId)->data->GetData(MEMORYDEVICE_CUDA);
	args.intensity_current = projectedIntensityHierarchy->GetLevel(levelId)->data->GetData(MEMORYDEVICE_CUDA);
	args.intensity_reference = viewHierarchy_intensity->GetLevel(levelId)->intensity_prev->GetData(MEMORYDEVICE_CUDA);
	args.gradient_reference = viewHierarchy_intensity->GetLevel(levelId)->gradients->GetData(MEMORYDEVICE_CUDA);
	args.deltaT = deltaT;
	args.intrinsics_depth = viewHierarchy_depth->GetLevel(levelId)->intrinsics;
	args.imgSize_depth = viewHierarchy_depth->GetLevel(levelId)->data->noDims;
	args.intrinsics_rgb = viewHierarchy_intensity->GetLevel(levelId)->intrinsics;
	args.imgSize_rgb = viewHierarchy_intensity->GetLevel(levelId)->intensity_prev->noDims;
	args.T_ref_CW = depthToRGBTransform * intensityReferencePose;
	args.T_scene_WC = renderedScenePose.inv();
	args.viewFrustum_max = 10; // FIXME: use parameter
	args.intensityThresh = colourThresh[levelId];
	args.minGradient = parameters.minColourGradient;

	RGBTrackerErrorImage_device << < gridSize, computationBlockSize >> >(image_out->GetData(MEMORYDEVICE_CUDA), args, 0.05);
	ORcudaKernelCheck;
}

__global__
void
computeDepthPointAndColour_device(Vector4f* out_points, float* out_rgb, const float* in_rgb, const float* in_points,
                                  Vector2i imageSize, Vector2i sceneSize, Vector4f intrinsics_depth,
                                  Vector4f intrinsics_rgb, Matrix4f T_depthToRGB)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	computeDepthPointAndColour(x, y, out_points, out_rgb, in_rgb, in_points, imageSize, sceneSize, intrinsics_depth,
	                           intrinsics_rgb, T_depthToRGB);
}

void ITMLib::ITMICPTracker_CUDA::ComputeDepthPointAndIntensity(ITMFloat4Image* points_out, ITMFloatImage* intensity_out,
                                                               const ITMFloatImage* intensity_in,
                                                               const ITMFloatImage* depth_in,
                                                               const Vector4f& intrinsics_depth,
                                                               const Vector4f& intrinsics_rgb,
                                                               const Matrix4f& T_depthToRGB)
{
	const Vector2i imageSize_rgb = intensity_in->noDims;
	const Vector2i imageSize_depth = depth_in->noDims; // Also the size of the projected image

	points_out->ChangeDims(imageSize_depth); // Actual reallocation should happen only once per run.
	intensity_out->ChangeDims(imageSize_depth); // Actual reallocation should happen only once per run.

	const float* depths = depth_in->GetData(MEMORYDEVICE_CUDA);
	const float* intensityIn = intensity_in->GetData(MEMORYDEVICE_CUDA);
	Vector4f* pointsOut = points_out->GetData(MEMORYDEVICE_CUDA);
	float* intensityOut = intensity_out->GetData(MEMORYDEVICE_CUDA);

	dim3 gridSize((int) std::ceil((float) imageSize_depth.x / (float) computationBlockSize.x),
	              (int) std::ceil((float) imageSize_depth.y / (float) computationBlockSize.y));

	computeDepthPointAndColour_device<<<gridSize, computationBlockSize>>>(pointsOut, intensityOut, intensityIn, depths,
	                                                           imageSize_rgb, imageSize_depth, intrinsics_rgb,
	                                                           intrinsics_depth, T_depthToRGB);
	ORcudaKernelCheck;
}

/**
 * @param h in [0, 360]
 * @param s in [0, 1]
 * @param v in [0, 1]
 * @return
 */
_CPU_AND_GPU_CODE_ inline
Vector3f HSVtoRGB(const float h, const float s, const float v)
{
	int sector = static_cast<int>(floor(h / 60));
	float f = (h / 60 - sector);
	float p = v * (1 - s);
	float q = v * (1 - s * f);
	float t = v * (1 - s * (1 - f));
	switch (sector)
	{
		case 0:
		case 6:
			return Vector3f(v, t, p);
		case 1:
			return Vector3f(q, v, p);
		case 2:
			return Vector3f(p, v, t);
		case 3:
			return Vector3f(p, q, v);
		case 4:
			return Vector3f(t, p, v);
		case 5:
			return Vector3f(v, p, q);
	}

	return Vector3f(0, 0, 0);
}

__device__ void
RGBTrackerErrorImage_device_main(Vector4u* outRendering,
                                 const Matrix4f& deltaT, const Matrix4f& T_ref_CW, const Matrix4f& T_scene_WC,
                                 const Vector4f* points_current,
                                 const float* intensity_current,
                                 const float* intensity_reference,
                                 const Vector2f* gradient_reference,
                                 const Vector4f& intrinsics_depth,
                                 const Vector4f& intrinsics_rgb,
                                 Vector2i imgSize_depth,
                                 Vector2i imgSize_rgb,
                                 float viewFrustum_max,
                                 float intensityThresh,
                                 float minGradient,
                                 float maxError)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	int threadId = threadIdx.x + threadIdx.y * blockDim.x;
	size_t blockId = blockIdx.x + blockIdx.y * gridDim.x;

	__shared__ bool blockHasValidPoint;

	blockHasValidPoint = false;
	__syncthreads();

	float A[6];
	float b;
	float weight;
	bool isValidPoint = false;

	for (int i = 0; i < 6; i++) A[i] = 0.0f;

	if (x >= imgSize_depth.x || y >= imgSize_depth.y)
		return;

	isValidPoint = computePerPointGH_RGB_Ab<false, false>(A, b, weight, x, y,
	                                                      deltaT,
	                                                      T_ref_CW,
	                                                      T_scene_WC,
	                                                      points_current,
	                                                      intensity_current,
	                                                      intensity_reference,
	                                                      gradient_reference,
	                                                      imgSize_depth,
	                                                      imgSize_rgb,
	                                                      intrinsics_depth,
	                                                      intrinsics_rgb,
	                                                      viewFrustum_max,
	                                                      intensityThresh,
	                                                      minGradient
	);

	int locId = PixelCoordsToIndex(x, y, imgSize_depth);
	if (!isValidPoint)
	{
		outRendering[locId] = Vector4u(0, 0, 0, 0);
		return;
	}

	b = MIN(fabs(b / maxError), 1); // normalize

	Vector4f color = Vector4f(HSVtoRGB((1 - b) * 240, 1, 1) * 255, 255);

	outRendering[locId] = color.toUChar();
}

__global__ void
RGBTrackerErrorImage_device(Vector4u* outRendering,
                            RGBKernelParameters para,
                            float maxError)
{
	RGBTrackerErrorImage_device_main(outRendering,
	                                 para.deltaT,
	                                 para.T_ref_CW,
	                                 para.T_scene_WC,
	                                 para.points_current,
	                                 para.intensity_current,
	                                 para.intensity_reference,
	                                 para.gradient_reference,
	                                 para.intrinsics_depth,
	                                 para.intrinsics_rgb,
	                                 para.imgSize_depth,
	                                 para.imgSize_rgb,
	                                 para.viewFrustum_max,
	                                 para.intensityThresh,
	                                 para.minGradient,
	                                 maxError);
}


// device functions

template<bool shortIteration, bool rotationOnly>
__device__ void
RGBTrackerOneLevel_g_rt_device_main(ITMICPTracker_CUDA::AccuCell* accu,
                                    const Matrix4f& deltaT, const Matrix4f& T_ref_CW, const Matrix4f& T_scene_WC,
                                    const Vector4f* points_current,
                                    const float* intensity_current,
                                    const float* intensity_reference,
                                    const Vector2f* gradient_reference,
                                    const Vector4f& intrinsics_depth,
                                    const Vector4f& intrinsics_rgb,
                                    Vector2i imgSize_depth,
                                    Vector2i imgSize_rgb,
                                    float viewFrustum_max,
                                    float intensityThresh,
                                    float minGradient)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	int threadId = threadIdx.x + threadIdx.y * blockDim.x;
	size_t blockId = blockIdx.x + blockIdx.y * gridDim.x;

	__shared__ bool blockHasValidPoint;

	blockHasValidPoint = false;
	__syncthreads();

	const int noPara = shortIteration ? 3 : 6;
	const int noParaSQ = shortIteration ? 3 + 2 + 1 : 6 + 5 + 4 + 3 + 2 + 1;
	float A[noPara];
	float b;
	float weight;
	bool isValidPoint = false;

	for (int i = 0; i < noPara; i++) A[i] = 0.0f;

	if (x < imgSize_depth.x && y < imgSize_depth.y)
	{
		isValidPoint = computePerPointGH_RGB_Ab<shortIteration, rotationOnly>(A, b, weight, x, y,
		                                                                      deltaT,
		                                                                      T_ref_CW,
		                                                                      T_scene_WC,
		                                                                      points_current,
		                                                                      intensity_current,
		                                                                      intensity_reference,
		                                                                      gradient_reference,
		                                                                      imgSize_depth,
		                                                                      imgSize_rgb,
		                                                                      intrinsics_depth,
		                                                                      intrinsics_rgb,
		                                                                      viewFrustum_max,
		                                                                      intensityThresh,
		                                                                      minGradient
		);
		if (isValidPoint) blockHasValidPoint = true;
	}

	if (!isValidPoint)
	{
		for (int i = 0; i < noPara; i++) A[i] = 0.0f;
		b = 0.0f;
		weight = 0.0f;
	}

	__syncthreads();

	if (!blockHasValidPoint) return;

	L2Loss loss;
//	HuberLoss loss(0.001);
//	TukeyLoss loss(0.001);
//	CauchyLoss loss(0.001);

	float lossWeight = loss.Weight(b);

	// reduction for valid number of points
	parallelReduce<BlockSideLength * BlockSideLength>(accu[blockId].numPoints, (int) isValidPoint, threadId);

	// reduction for error
	parallelReduce<BlockSideLength * BlockSideLength>(accu[blockId].f, weight * loss.Loss(b), threadId);

	// reduction for nabla (b * A)
	for (unsigned char paraId = 0; paraId < noPara; paraId += 3)
	{
		parallelReduceArray3<BlockSideLength * BlockSideLength>(accu[blockId].g + paraId,
		                          (lossWeight * weight * b * Vector3f(A[paraId + 0], A[paraId + 1], A[paraId + 2])).v,
		                          threadId);
	}

	float localHessian[noParaSQ];
#if (defined(__CUDACC__) && defined(__CUDA_ARCH__))
#pragma unroll
#endif
	for (unsigned char r = 0, counter = 0; r < noPara; r++)
	{
#if (defined(__CUDACC__) && defined(__CUDA_ARCH__))
#pragma unroll
#endif
		for (int c = 0; c <= r; c++, counter++) localHessian[counter] = lossWeight * A[r] * A[c];
	}

	//reduction for hessian
	for (unsigned char paraId = 0; paraId < noParaSQ; paraId += 3)
	{
		parallelReduceArray3<BlockSideLength * BlockSideLength>(accu[blockId].h + paraId,
		                          localHessian + paraId,
		                          threadId);
	}
}

template<bool shortIteration, bool rotationOnly>
__device__ void
depthTrackerOneLevel_g_rt_device_main(ITMICPTracker_CUDA::AccuCell* accu,
                                      const Matrix4f& deltaT, const Matrix4f& T_scene_CW,
                                      const float* depth, const Vector4f* pointsMap, const Vector4f* normalsMap,
                                      const Vector4f sceneIntrinsics, const Vector2i sceneImageSize,
                                      Vector4f viewIntrinsics, Vector2i viewImageSize, float distThresh)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	int threadId = threadIdx.x + threadIdx.y * blockDim.x;
	size_t blockId = blockIdx.x + blockIdx.y * gridDim.x;

	__shared__ bool blockHasValidPoint;

	blockHasValidPoint = false;
	__syncthreads();

	const int noPara = shortIteration ? 3 : 6;
	const int noParaSQ = shortIteration ? 3 + 2 + 1 : 6 + 5 + 4 + 3 + 2 + 1;
	float A[noPara];
	float b;
	float weight;
	bool isValidPoint = false;

	if (x < viewImageSize.x && y < viewImageSize.y)
	{
		isValidPoint = computePerPointGH_Depth_Ab<shortIteration, rotationOnly>(A, b, weight, x, y,
		                                                                        deltaT, T_scene_CW,
		                                                                        depth, viewImageSize, viewIntrinsics,
		                                                                        sceneImageSize, sceneIntrinsics,
		                                                                        pointsMap,
		                                                                        normalsMap, distThresh);
		if (isValidPoint) blockHasValidPoint = true;
	}

	__syncthreads();

	if (!blockHasValidPoint) return;

	if (!isValidPoint)
	{
		for (int i = 0; i < noPara; i++) A[i] = 0.0f;
		b = 0.0f;
		weight = 0.0f;
	}

	L2Loss loss;
//	HuberLoss loss(0.001);
//	TukeyLoss loss(0.001);
//	CauchyLoss loss(0.001);

	float lossWeight = loss.Weight(b);

	// reduction for valid number of points
	parallelReduce<BlockSideLength * BlockSideLength>(accu[blockId].numPoints, (int) isValidPoint, threadId);

	// reduction for error
	parallelReduce<BlockSideLength * BlockSideLength>(accu[blockId].f, weight * loss.Loss(b), threadId);

	// reduction for nabla (b * A)
	for (unsigned char paraId = 0; paraId < noPara; paraId += 3)
	{
		parallelReduceArray3<BlockSideLength * BlockSideLength>(accu[blockId].g + paraId,
		                          (lossWeight * weight * b * Vector3f(A[paraId + 0], A[paraId + 1], A[paraId + 2])).v,
		                          threadId);
	}

	float localHessian[noParaSQ];
#if (defined(__CUDACC__) && defined(__CUDA_ARCH__))
#pragma unroll
#endif
	for (unsigned char r = 0, counter = 0; r < noPara; r++)
	{
#if (defined(__CUDACC__) && defined(__CUDA_ARCH__))
#pragma unroll
#endif
		for (int c = 0; c <= r; c++, counter++) localHessian[counter] = lossWeight * weight * A[r] * A[c];
	}

	//reduction for hessian
	for (unsigned char paraId = 0; paraId < noParaSQ; paraId += 3)
	{
		parallelReduceArray3<BlockSideLength * BlockSideLength>(accu[blockId].h + paraId,
		                          localHessian + paraId,
		                          threadId);
	}
}

__global__ void
Sim3DerivativeDepth_device(AccumulatorSim3* accumulator,
                           const float* depth, const Vector4f* pointsMap, const Vector4f* normalsMap,
                           Vector4f sceneIntrinsics, Vector2i sceneImageSize,
                           Vector4f viewIntrinsics, Vector2i viewImageSize,
                           Matrix4f deltaT, Matrix4f T_scene_CW,
                           const float distThresh)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	int threadId = threadIdx.x + threadIdx.y * blockDim.x;
	size_t blockId = blockIdx.x + blockIdx.y * gridDim.x;

	__shared__ bool blockHasValidPoint;

	blockHasValidPoint = false;
	__syncthreads();

	MatrixXf<1, 7> J;
	float r;
	float f;
	float weight;
	bool isValidPoint = false;
	if (x < viewImageSize.x && y < viewImageSize.y)
	{
		isValidPoint = computeSim3Derivative_Depth(J, r, weight, x, y,
		                                           depth, viewImageSize,
		                                           viewIntrinsics, sceneImageSize, sceneIntrinsics,
		                                           deltaT, T_scene_CW,
		                                           pointsMap, normalsMap, distThresh);

		if (isValidPoint) blockHasValidPoint = true;
	}

	__syncthreads();

	if (!blockHasValidPoint)
		return;

	if (!isValidPoint)
	{
		J.setZeros();
		f = 0;
		r = 0;
		weight = 1;
	}

	parallelReduce<BlockSideLength * BlockSideLength>(accumulator[blockId].numPoints, (int) isValidPoint, threadId);
	parallelReduce<BlockSideLength * BlockSideLength>(accumulator[blockId].f, weight * r * r, threadId);

	MatrixXf<1, 7> localG = weight * (J * r);
	parallelReduceArray4<BlockSideLength * BlockSideLength>(accumulator[blockId].g, localG.m, threadId);
	parallelReduceArray3<BlockSideLength * BlockSideLength>(accumulator[blockId].g + 4, localG.m + 4, threadId);

	const int sizeH = 7 + 6 + 5 + 4 + 3 + 2 + 1; // size of lower triangle matrix
	float localHessian[sizeH];
#pragma unroll
	for (unsigned char row = 0, counter = 0; row < 7; row++)
	{
#pragma unroll
		for (int col = 0; col <= row; col++, counter++) localHessian[counter] = weight * J.at(row, 0) * J.at(col, 0);
	}

	//reduction for hessian
	for (unsigned char offset = 0; offset < sizeH; offset += 4)
	{
		parallelReduceArray4<BlockSideLength * BlockSideLength>(accumulator[blockId].H + offset,
		                          localHessian + offset,
		                          threadId);
	}
}

__global__ void
Sim3DerivativeRGB_device(AccumulatorSim3* accumulator,
                         Vector4f* points_current,
                         float* intensity_current, float* intensity_reference, Vector2f* gradient_reference,
                         Vector4f intrinsics_depth, Vector4f intrinsics_rgb,
                         Vector2i imgSize_depth, Vector2i imgSize_rgb,
                         const Matrix4f deltaT, const Matrix4f T_ref_CW, const Matrix4f T_scene_WC,
                         const float viewFrustum_max, const float intensityThresh, const float minGradient)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	int threadId = threadIdx.x + threadIdx.y * blockDim.x;
	size_t blockId = blockIdx.x + blockIdx.y * gridDim.x;

	__shared__ bool blockHasValidPoint;

	blockHasValidPoint = false;
	__syncthreads();

	MatrixXf<1, 7> J;
	float r;
	float f;
	float weight;
	bool isValidPoint = false;
	if (x < imgSize_depth.x && y < imgSize_depth.y)
	{
		isValidPoint = computeSim3Derivative_RGB(J, r, weight, x, y,
		                                         points_current,
		                                         intensity_current, intensity_reference, gradient_reference,
		                                         imgSize_depth, imgSize_rgb,
		                                         intrinsics_depth, intrinsics_rgb,
		                                         deltaT, T_ref_CW, T_scene_WC,
		                                         viewFrustum_max, intensityThresh, minGradient);

		if (isValidPoint) blockHasValidPoint = true;
	}

	__syncthreads();

	if (!blockHasValidPoint)
		return;

	if (!isValidPoint)
	{
		J.setZeros();
		f = 0;
		r = 0;
		weight = 1;
	}

	parallelReduce<BlockSideLength * BlockSideLength>(accumulator[blockId].numPoints, (int) isValidPoint, threadId);
	parallelReduce<BlockSideLength * BlockSideLength>(accumulator[blockId].f, weight * r * r, threadId);

	MatrixXf<1, 7> localG = weight * (J * r);
	parallelReduceArray4<BlockSideLength * BlockSideLength>(accumulator[blockId].g, localG.m, threadId);
	parallelReduceArray3<BlockSideLength * BlockSideLength>(accumulator[blockId].g + 4, localG.m + 4, threadId);

	const int sizeH = 7 + 6 + 5 + 4 + 3 + 2 + 1; // size of lower triangle matrix
	float localHessian[sizeH];
#pragma unroll
	for (unsigned char row = 0, counter = 0; row < 7; row++)
	{
#pragma unroll
		for (int col = 0; col <= row; col++, counter++) localHessian[counter] = weight * J.at(row, 0) * J.at(col, 0);
	}

	//reduction for hessian
	for (unsigned char offset = 0; offset < sizeH; offset += 4)
	{
		parallelReduceArray4<BlockSideLength * BlockSideLength>(accumulator[blockId].H + offset,
		                          localHessian + offset,
		                          threadId);
	}
}

template<bool shortIteration, bool rotationOnly>
__global__ void
depthTrackerOneLevel_g_rt_device(KernelParameters para)
{
	depthTrackerOneLevel_g_rt_device_main<shortIteration, rotationOnly>(para.accu,
	                                                                    para.deltaT, para.T_scene_CW,
	                                                                    para.depth, para.pointsMap, para.normalsMap,
	                                                                    para.sceneIntrinsics, para.sceneImageSize,
	                                                                    para.viewIntrinsics,
	                                                                    para.viewImageSize,
	                                                                    para.distThresh);
}

template<bool shortIteration, bool rotationOnly>
__global__ void RGBTrackerOneLevel_g_rt_device(RGBKernelParameters para)
{
	RGBTrackerOneLevel_g_rt_device_main<shortIteration, rotationOnly>(para.accu,
	                                                                  para.deltaT,
	                                                                  para.T_ref_CW,
	                                                                  para.T_scene_WC,
	                                                                  para.points_current,
	                                                                  para.intensity_current,
	                                                                  para.intensity_reference,
	                                                                  para.gradient_reference,
	                                                                  para.intrinsics_depth,
	                                                                  para.intrinsics_rgb,
	                                                                  para.imgSize_depth,
	                                                                  para.imgSize_rgb,
	                                                                  para.viewFrustum_max,
	                                                                  para.intensityThresh,
	                                                                  para.minGradient);
}