// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMICPTracker_CUDA.h"
#include "../Shared/ITMICPTracker_Shared.h"
#include <Trackers/Shared/ITMTracker_Shared.h>
#include <Utils/ITMCUDAUtils.h>
#include <ORUtils/CUDADefines.h>
#include <ORUtils/EigenConversion.h>

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

struct AccumulatorSahillioglu
{
	int numPoints;
	float a;
	float b;
	Vector3f c;
	Vector3f d;
	float f; // error
};

struct AccumulatorTransScale
{
	int numPoints;
	Matrix4f H; // hessian
	Vector4f g;
	float f; // error

	_CPU_AND_GPU_CODE_
	void SetZero()
	{
		numPoints = 0;
		f = 0;
		H.setZeros();
#pragma unroll
		for (float& i : g.v)
			i = 0;
	}
};

struct KernelParameters
{
	ITMICPTracker_CUDA::AccuCell* accu;
	float* depth;
	Matrix4f approxInvPose;
	Vector4f* pointsMap;
	Vector4f* normalsMap;
	Vector4f sceneIntrinsics;
	Vector2i sceneImageSize;
	Matrix4f scenePose;
	Vector4f viewIntrinsics;
	Vector2i viewImageSize;
	float scaleFactor;
	float distThresh;
};

struct RGBKernelParameters
{
	ITMICPTracker_CUDA::AccuCell* accu;
	Vector4f* points_current;
	float* intensity_current;
	float* intensity_reference;
	Vector2f* gradient_reference;
	Matrix4f approxInvPose;
	Matrix4f intensityReferencePose;
	Vector4f intrinsics_depth;
	Vector4f intrinsics_rgb;
	Vector2i imgSize_depth;
	Vector2i imgSize_rgb;
	float viewFrustum_max;
	float intensityThresh;
	float minGradient;
};

__global__ void
ScaleTracker_device(AccumulatorTransScale* accumulator,
                    const float* depth, const Vector4f* pointsMap, const Vector4f* normalsMap,
                    Vector4f sceneIntrinsics, Vector2i sceneImageSize,
                    Vector4f viewIntrinsics, Vector2i viewImageSize,
                    Matrix4f approxInvPose, Matrix4f scenePose, float scaleFactor, float distThresh);

__global__ void
TrackerSahillioglu_device(AccumulatorSahillioglu* accumulator,
                          const float* depth, const Vector4f* pointsMap,
                          const Vector4f sceneIntrinsics, const Vector2i sceneImageSize,
                          const Vector4f viewIntrinsics, const Vector2i viewImageSize,
                          const Matrix4f approxInvPose, const float approxScaleFactor, const Matrix4f scenePose,
                          const float distThresh);

template<bool shortIteration, bool rotationOnly>
__global__ void
depthTrackerOneLevel_g_rt_device(KernelParameters para);

template<bool shortIteration, bool rotationOnly>
__global__ void RGBTrackerOneLevel_g_rt_device(RGBKernelParameters para);

// host methods

ITMICPTracker_CUDA::ITMICPTracker_CUDA(Vector2i imgSize_d, Vector2i imgSize_rgb, const Parameters& parameters,
                                       const ITMLowLevelEngine* lowLevelEngine)
	: ITMICPTracker(imgSize_d, imgSize_rgb, parameters, lowLevelEngine, MEMORYDEVICE_CUDA)
{
	dim3 gridSize((int) ceil(imgSize_d.x / 16.0f),
	              (int) ceil(imgSize_d.y / 16.0f));
	size_t numBlocks = gridSize.x * gridSize.y;
	ORcudaSafeCall(cudaMalloc(&accumulator_device, numBlocks * sizeof(AccuCell)));
}

ITMICPTracker_CUDA::~ITMICPTracker_CUDA(void)
{
	ORcudaSafeCall(cudaFree(accumulator_device));
}

size_t ITMICPTracker_CUDA::ComputeSahillioglu(float& f, Eigen::Matrix<EigenT, 4, 4>& A, Eigen::Matrix<EigenT, 4, 1>& b,
                                              const Matrix4f& approxInvPose, const float approxScaleFactor)
{
	Vector4f* pointsMap = sceneHierarchy->GetLevel(0)->pointsMap->GetData(MEMORYDEVICE_CUDA);
	Vector4f sceneIntrinsics = sceneHierarchy->GetLevel(0)->intrinsics;
	Vector2i sceneImageSize = sceneHierarchy->GetLevel(0)->pointsMap->noDims;

	float* depth = viewHierarchy_depth->GetLevel(levelId)->data->GetData(MEMORYDEVICE_CUDA);
	Vector4f viewIntrinsics = viewHierarchy_depth->GetLevel(levelId)->intrinsics;
	Vector2i viewImageSize = viewHierarchy_depth->GetLevel(levelId)->data->noDims;

	AccumulatorSahillioglu* accumulator_device;
	ORcudaSafeCall(cudaMalloc((void**) &accumulator_device, sizeof(AccumulatorSahillioglu)));
	ORcudaSafeCall(cudaMemset(accumulator_device, 0, sizeof(AccumulatorSahillioglu)));

	dim3 blockSize(16, 16);
	dim3 gridSize((int) ceil((float) viewImageSize.x / (float) blockSize.x),
	              (int) ceil((float) viewImageSize.y / (float) blockSize.y));

	Vector3f viewCloudCenter_world = (approxInvPose * Vector4f(depthPointCloudCenter, 1)).toVector3();

	TrackerSahillioglu_device<<<gridSize, blockSize>>>(accumulator_device,
	                                                   depth,
	                                                   pointsMap,
	                                                   sceneIntrinsics,
	                                                   sceneImageSize,
	                                                   viewIntrinsics,
	                                                   viewImageSize,
	                                                   approxInvPose,
	                                                   approxScaleFactor,
	                                                   renderedScenePose,
	                                                   distThresh[levelId]
	);
	ORcudaKernelCheck;

	AccumulatorSahillioglu accumulator_host;
	ORcudaSafeCall(cudaMemcpy(&accumulator_host, accumulator_device, sizeof(accumulator_host), cudaMemcpyDeviceToHost));

	if (accumulator_host.numPoints > 0)
	{
		accumulator_host.a /= accumulator_host.numPoints;
		accumulator_host.b /= accumulator_host.numPoints;
		accumulator_host.c /= accumulator_host.numPoints;
		accumulator_host.d /= accumulator_host.numPoints;
	}

	b << accumulator_host.b, accumulator_host.d.x, accumulator_host.d.y, accumulator_host.d.z;
	A.block<4, 1>(0, 0) << accumulator_host.a, accumulator_host.c.x, accumulator_host.c.y, accumulator_host.c.z;
	A.block<1, 3>(0, 1) << accumulator_host.c.x, accumulator_host.c.y, accumulator_host.c.z;
	A.block<3, 3>(1, 1).setIdentity();
//	A.block<3, 3>(1, 1).setZero();
//	A.block<3, 3>(1, 1).diagonal().setConstant(accumulator_host.numPoints);
	f = accumulator_host.f / accumulator_host.numPoints;

	ORcudaSafeCall(cudaFree(accumulator_device));
	return accumulator_host.numPoints;
}

/** Perform final reduction of per-block summation values stored in accumulator array
 * */
template<size_t blockSize>
__global__ void reduceAccumulatorTransScale(AccumulatorTransScale* accu, size_t N)
{
	size_t threadId = threadIdx.x + blockDim.x * threadIdx.y;
	size_t blockId = blockIdx.x + gridDim.x * blockIdx.y;

	AccumulatorTransScale sum{};
	sum.SetZero();
	for (size_t i = blockId * blockSize + threadId; i < N; i += blockDim.x * gridDim.x)
	{
		sum.numPoints += accu[i].numPoints;
		sum.f += accu[i].f;
		sum.g = accu[i].g;
		sum.H = accu[i].H;
	}

	parallelReduce<blockSize>(accu[blockId].numPoints, sum.numPoints, threadId);
	parallelReduce<blockSize>(accu[blockId].f, sum.f, threadId);
	parallelReduceArray4<blockSize>(accu[blockId].g.v, sum.g.v, threadId);
	for (unsigned char offset = 0; offset < 16; offset += 4)
		parallelReduceArray4<blockSize>(accu[blockId].H.m + offset, sum.H.m + offset, threadId);
}

size_t ITMICPTracker_CUDA::ComputeTransScale(float& f, Eigen::Matrix<EigenT, 4, 4>& H, Eigen::Matrix<EigenT, 4, 1>& g,
                                             const Matrix4f& approxInvPose, float approxScaleFactor)
{
	Vector4f* pointsMap = sceneHierarchy->GetLevel(0)->pointsMap->GetData(MEMORYDEVICE_CUDA);
	Vector4f* normalsMap = sceneHierarchy->GetLevel(0)->normalsMap->GetData(MEMORYDEVICE_CUDA);
	Vector4f sceneIntrinsics = sceneHierarchy->GetLevel(0)->intrinsics;
	Vector2i sceneImageSize = sceneHierarchy->GetLevel(0)->pointsMap->noDims;

	float* depth = viewHierarchy_depth->GetLevel(levelId)->data->GetData(MEMORYDEVICE_CUDA);
	Vector4f viewIntrinsics = viewHierarchy_depth->GetLevel(levelId)->intrinsics;
	Vector2i viewImageSize = viewHierarchy_depth->GetLevel(levelId)->data->noDims;

	dim3 blockSize(16, 16);
	dim3 gridSize((int) ceil((float) viewImageSize.x / (float) blockSize.x),
	              (int) ceil((float) viewImageSize.y / (float) blockSize.y));
	size_t numBlocks = gridSize.x * gridSize.y;

	AccumulatorTransScale* accumulator_device;
	ORcudaSafeCall(cudaMalloc((void**) &accumulator_device, numBlocks * sizeof(AccumulatorTransScale)));
	ORcudaSafeCall(cudaMemset(accumulator_device, 0, numBlocks * sizeof(AccumulatorTransScale)));

	ScaleTracker_device<<<gridSize, blockSize>>>(accumulator_device,
	                                             depth,
	                                             pointsMap,
	                                             normalsMap,
	                                             sceneIntrinsics,
	                                             sceneImageSize,
	                                             viewIntrinsics,
	                                             viewImageSize,
	                                             approxInvPose,
	                                             renderedScenePose,
	                                             approxScaleFactor,
	                                             distThresh[levelId]
	);
	reduceAccumulatorTransScale<1024><<<1, 1024>>>(accumulator_device, numBlocks);
	ORcudaKernelCheck;

	AccumulatorTransScale accumulator_host;
	ORcudaSafeCall(
		cudaMemcpy(&accumulator_host, accumulator_device, sizeof(AccumulatorTransScale), cudaMemcpyDeviceToHost));

	H = ORUtils::ToEigen<Eigen::Matrix<EigenT, 4, 4>>(accumulator_host.H);
	g = ORUtils::ToEigen<Eigen::Matrix<EigenT, 4, 1>>(accumulator_host.g);
	f = accumulator_host.f;
	if (accumulator_host.numPoints)
	{
		H /= accumulator_host.numPoints;
		g /= accumulator_host.numPoints;
		f /= accumulator_host.numPoints;
	}

	ORcudaSafeCall(cudaFree(accumulator_device));
	return accumulator_host.numPoints;
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

int ITMICPTracker_CUDA::ComputeGandH_Depth(float& f, float* nabla, float* hessian, Matrix4f approxInvPose,
                                           float approxScaleFactor)
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

	dim3 blockSize(16, 16);
	dim3 gridSize((int) ceil((float) viewImageSize.x / (float) blockSize.x),
	              (int) ceil((float) viewImageSize.y / (float) blockSize.y));

	size_t numBlocks = gridSize.x * gridSize.y;
	ORcudaSafeCall(cudaMemset(accumulator_device, 0, numBlocks * sizeof(AccuCell)));
	struct KernelParameters args;
	args.accu = accumulator_device;
	args.depth = depth;
	args.approxInvPose = approxInvPose;
	args.pointsMap = pointsMap;
	args.normalsMap = normalsMap;
	args.sceneIntrinsics = sceneIntrinsics;
	args.sceneImageSize = sceneImageSize;
	args.scenePose = renderedScenePose;
	args.viewIntrinsics = viewIntrinsics;
	args.viewImageSize = viewImageSize;
	args.scaleFactor = approxScaleFactor;
	args.distThresh = distThresh[levelId];

	switch (iterationType)
	{
		case TRACKER_ITERATION_ROTATION:
			depthTrackerOneLevel_g_rt_device<true, true> << < gridSize, blockSize >> >(args);
			reduceAccumulator<true, 1024><<<1, 1024>>>(accumulator_device, numBlocks);
			ORcudaKernelCheck;
			break;
		case TRACKER_ITERATION_TRANSLATION:
			depthTrackerOneLevel_g_rt_device<true, false> << < gridSize, blockSize >> >(args);
			reduceAccumulator<true, 1024><<<1, 1024>>>(accumulator_device, numBlocks);
			ORcudaKernelCheck;
			break;
		case TRACKER_ITERATION_BOTH:
			depthTrackerOneLevel_g_rt_device<false, false> << < gridSize, blockSize >> >(args);
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

int ITMICPTracker_CUDA::ComputeGandH_RGB(float& f, float* nabla, float* hessian, Matrix4f approxInvPose,
                                         float approxScaleFactor)
{
	Vector2i imageSize_depth = viewHierarchy_depth->GetLevel(levelId)->data->noDims;
	Vector2i imageSize_rgb = viewHierarchy_depth->GetLevel(levelId)->data->noDims;

	if (iterationType == TRACKER_ITERATION_NONE) return 0;

	bool shortIteration = iterationType == TRACKER_ITERATION_ROTATION
	                      || iterationType == TRACKER_ITERATION_TRANSLATION;
	int noPara = shortIteration ? 3 : 6;

	dim3 blockSize(16, 16);
	dim3 gridSize((int) ceil((float) imageSize_depth.x / (float) blockSize.x),
	              (int) ceil((float) imageSize_depth.y / (float) blockSize.y));

	ORcudaSafeCall(cudaMemset(accumulator_device, 0, sizeof(AccuCell)));

	struct RGBKernelParameters args;
	args.accu = accumulator_device;
	args.points_current = reprojectedPointsHierarchy->GetLevel(levelId)->data->GetData(MEMORYDEVICE_CUDA);
	args.intensity_current = projectedIntensityHierarchy->GetLevel(levelId)->data->GetData(MEMORYDEVICE_CUDA);
	args.intensity_reference = viewHierarchy_intensity->GetLevel(levelId)->intensity_prev->GetData(MEMORYDEVICE_CUDA);
	args.gradient_reference = viewHierarchy_intensity->GetLevel(levelId)->gradients->GetData(MEMORYDEVICE_CUDA);
	args.approxInvPose = approxInvPose;
	args.intrinsics_depth = viewHierarchy_depth->GetLevel(levelId)->intrinsics;
	args.imgSize_depth = viewHierarchy_depth->GetLevel(levelId)->data->noDims;
	args.intrinsics_rgb = viewHierarchy_intensity->GetLevel(levelId)->intrinsics;
	args.imgSize_rgb = viewHierarchy_intensity->GetLevel(levelId)->intensity_prev->noDims;
	args.intensityReferencePose = depthToRGBTransform * intensityReferencePose;
	args.viewFrustum_max = 6; // FIXME: use parameter
	args.intensityThresh = colourThresh[levelId];
	args.minGradient = parameters.minColourGradient;

	switch (iterationType)
	{
		case TRACKER_ITERATION_ROTATION:
			RGBTrackerOneLevel_g_rt_device<true, true> << < gridSize, blockSize >> >(args);
			ORcudaKernelCheck;
			break;
		case TRACKER_ITERATION_TRANSLATION:
			RGBTrackerOneLevel_g_rt_device<true, false> << < gridSize, blockSize >> >(args);
			ORcudaKernelCheck;
			break;
		case TRACKER_ITERATION_BOTH:
			RGBTrackerOneLevel_g_rt_device<false, false> << < gridSize, blockSize >> >(args);
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

__global__
void
computeDepthPointAndColour_device(Vector4f* out_points, float* out_rgb, const float* in_rgb, const float* in_points,
                                  Vector2i imageSize, Vector2i sceneSize, Vector4f intrinsics_depth,
                                  Vector4f intrinsics_rgb, Matrix4f scenePose)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	computeDepthPointAndColour(x, y, out_points, out_rgb, in_rgb, in_points, imageSize, sceneSize, intrinsics_depth,
	                           intrinsics_rgb, scenePose);
}

void ITMLib::ITMICPTracker_CUDA::ComputeDepthPointAndIntensity(ITMFloat4Image* points_out, ITMFloatImage* intensity_out,
                                                               const ITMFloatImage* intensity_in,
                                                               const ITMFloatImage* depth_in,
                                                               const Vector4f& intrinsics_depth,
                                                               const Vector4f& intrinsics_rgb,
                                                               const Matrix4f& scenePose)
{
	const Vector2i imageSize_rgb = intensity_in->noDims;
	const Vector2i imageSize_depth = depth_in->noDims; // Also the size of the projected image

	points_out->ChangeDims(imageSize_depth); // Actual reallocation should happen only once per run.
	intensity_out->ChangeDims(imageSize_depth); // Actual reallocation should happen only once per run.

	const float* depths = depth_in->GetData(MEMORYDEVICE_CUDA);
	const float* intensityIn = intensity_in->GetData(MEMORYDEVICE_CUDA);
	Vector4f* pointsOut = points_out->GetData(MEMORYDEVICE_CUDA);
	float* intensityOut = intensity_out->GetData(MEMORYDEVICE_CUDA);

	dim3 blockSize(16, 16);
	dim3 gridSize((int) ceil((float) imageSize_depth.x / (float) blockSize.x),
	              (int) ceil((float) imageSize_depth.y / (float) blockSize.y));

	computeDepthPointAndColour_device<<<gridSize, blockSize>>>(pointsOut, intensityOut, intensityIn, depths,
	                                                           imageSize_rgb, imageSize_depth, intrinsics_rgb,
	                                                           intrinsics_depth, scenePose);
	ORcudaKernelCheck;
}

// device functions

template<bool shortIteration, bool rotationOnly>
__device__ void
RGBTrackerOneLevel_g_rt_device_main(ITMICPTracker_CUDA::AccuCell* accu,
                                    Vector4f* points_current,
                                    float* intensity_current,
                                    float* intensity_reference,
                                    Vector2f* gradient_reference,
                                    Matrix4f approxInvPose,
                                    Matrix4f intensityReferencePose,
                                    Vector4f intrinsics_depth,
                                    Vector4f intrinsics_rgb,
                                    Vector2i imgSize_depth,
                                    Vector2i imgSize_rgb,
                                    float viewFrustum_max,
                                    float intensityThresh,
                                    float minGradient)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	int locId_local = threadIdx.x + threadIdx.y * blockDim.x;

	__shared__ bool blockHasValidPoint;

	blockHasValidPoint = false;
	__syncthreads();

	const int noPara = shortIteration ? 3 : 6;
	const int noParaSQ = shortIteration ? 3 + 2 + 1 : 6 + 5 + 4 + 3 + 2 + 1;
	float A[noPara];
	float b;
	bool isValidPoint = false;

	for (int i = 0; i < noPara; i++) A[i] = 0.0f;

	if (x < imgSize_depth.x && y < imgSize_depth.y)
	{
		isValidPoint = computePerPointGH_RGB_Ab<shortIteration, rotationOnly>(A, b, x, y,
		                                                                      points_current,
		                                                                      intensity_current,
		                                                                      intensity_reference,
		                                                                      gradient_reference,
		                                                                      imgSize_depth,
		                                                                      imgSize_rgb,
		                                                                      intrinsics_depth,
		                                                                      intrinsics_rgb,
		                                                                      approxInvPose,
		                                                                      intensityReferencePose,
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
	}

	__syncthreads();

	if (!blockHasValidPoint) return;

	// reduction for valid number of points
	parallelReduceAtomic<256>(accu->numPoints, (int) isValidPoint, locId_local);

	// reduction for error
	parallelReduceAtomic<256>(accu->f, b * b, locId_local);

	// reduction for nabla (b * A)
	for (unsigned char paraId = 0; paraId < noPara; paraId += 3)
	{
		parallelReduceArray3Atomic<256>(accu->g + paraId,
		                                Vector3f(b * A[paraId + 0], b * A[paraId + 1], b * A[paraId + 2]).v,
		                                locId_local);
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
		for (int c = 0; c <= r; c++, counter++) localHessian[counter] = A[r] * A[c];
	}

	//reduction for hessian
	for (unsigned char paraId = 0; paraId < noParaSQ; paraId += 3)
	{
		parallelReduceArray3Atomic<256>(accu->h + paraId,
		                                localHessian + paraId,
		                                locId_local);
	}
}

template<bool shortIteration, bool rotationOnly>
__device__ void
depthTrackerOneLevel_g_rt_device_main(ITMICPTracker_CUDA::AccuCell* accu, float* depth, Matrix4f approxInvPose,
                                      Vector4f* pointsMap,
                                      Vector4f* normalsMap, Vector4f sceneIntrinsics, Vector2i sceneImageSize,
                                      Matrix4f scenePose, Vector4f viewIntrinsics, Vector2i viewImageSize,
                                      float scaleFactor, float distThresh)
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
	bool isValidPoint = false;

	if (x < viewImageSize.x && y < viewImageSize.y)
	{
		isValidPoint = computePerPointGH_Depth_Ab<shortIteration, rotationOnly>(A, b, x, y, depth,
		                                                                        viewImageSize, viewIntrinsics,
		                                                                        sceneImageSize, sceneIntrinsics,
		                                                                        approxInvPose, scenePose, pointsMap,
		                                                                        normalsMap, scaleFactor, distThresh);
		if (isValidPoint) blockHasValidPoint = true;
	}

	__syncthreads();

	if (!blockHasValidPoint) return;

	if (!isValidPoint)
	{
		for (int i = 0; i < noPara; i++) A[i] = 0.0f;
		b = 0.0f;
	}

	// reduction for valid number of points
	parallelReduce<256>(accu[blockId].numPoints, (int) isValidPoint, threadId);

	// reduction for error
	parallelReduce<256>(accu[blockId].f, b * b, threadId);

	// reduction for nabla (b * A)
	for (unsigned char paraId = 0; paraId < noPara; paraId += 3)
	{
		parallelReduceArray3<256>(accu[blockId].g + paraId,
		                          Vector3f(b * A[paraId + 0], b * A[paraId + 1], b * A[paraId + 2]).v,
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
		for (int c = 0; c <= r; c++, counter++) localHessian[counter] = A[r] * A[c];
	}

	//reduction for hessian
	for (unsigned char paraId = 0; paraId < noParaSQ; paraId += 3)
	{
		parallelReduceArray3<256>(accu[blockId].h + paraId,
		                          localHessian + paraId,
		                          threadId);
	}
}

__global__ void
TrackerSahillioglu_device(AccumulatorSahillioglu* accumulator,
                          const float* depth, const Vector4f* pointsMap,
                          const Vector4f sceneIntrinsics, const Vector2i sceneImageSize,
                          const Vector4f viewIntrinsics, const Vector2i viewImageSize,
                          const Matrix4f approxInvPose, const float approxScaleFactor, const Matrix4f scenePose,
                          const float distThresh
)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	int locId_local = threadIdx.x + threadIdx.y * blockDim.x;

	__shared__ bool blockHasValidPoint;

	blockHasValidPoint = false;
	__syncthreads();

	Vector3f c, d;
	float a, b, f;
	bool isValidPoint = false;
	if (x < viewImageSize.x && y < viewImageSize.y)
	{
		isValidPoint = computePerPoint_Sahillioglu(f, a, b, c, d,
		                                           x, y, depth, viewImageSize,
		                                           viewIntrinsics, pointsMap, sceneImageSize, sceneIntrinsics,
		                                           approxInvPose, approxScaleFactor, scenePose, distThresh);

		if (isValidPoint) blockHasValidPoint = true;
	}

	__syncthreads();

	if (!blockHasValidPoint)
		return;

	if (!isValidPoint)
	{
		f = 0;
		a = 0;
		b = 0;
		memset(c.v, 0, sizeof(c));
		memset(d.v, 0, sizeof(d));
	}

	parallelReduceAtomic<256>(accumulator->numPoints, (int) isValidPoint, locId_local);
	parallelReduceAtomic<256>(accumulator->f, f, locId_local);
	parallelReduceAtomic<256>(accumulator->a, a, locId_local);
	parallelReduceAtomic<256>(accumulator->b, b, locId_local);
	parallelReduceArray3Atomic<256>(accumulator->c.v, c.v, locId_local);
	parallelReduceArray3Atomic<256>(accumulator->d.v, d.v, locId_local);
}

__global__ void
ScaleTracker_device(AccumulatorTransScale* accumulator,
                    const float* depth, const Vector4f* pointsMap, const Vector4f* normalsMap,
                    const Vector4f sceneIntrinsics, const Vector2i sceneImageSize,
                    const Vector4f viewIntrinsics, const Vector2i viewImageSize,
                    const Matrix4f approxInvPose, const Matrix4f scenePose, const float scaleFactor,
                    const float distThresh
)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	int threadId = threadIdx.x + threadIdx.y * blockDim.x;
	size_t blockId = blockIdx.x + blockIdx.y * gridDim.x;

	__shared__ bool blockHasValidPoint;

	blockHasValidPoint = false;
	__syncthreads();

	Vector4f j;
	float r;
	float f;
	bool isValidPoint = false;
	if (x < viewImageSize.x && y < viewImageSize.y)
	{
		isValidPoint = computePerPoint_TransScale(f, j, r, x, y,
		                                          depth, viewImageSize,
		                                          viewIntrinsics, pointsMap, normalsMap, sceneImageSize, sceneIntrinsics,
		                                          approxInvPose, scenePose, scaleFactor, distThresh);

		if (isValidPoint) blockHasValidPoint = true;
	}

	if (!isValidPoint)
	{
		memset(j.v, 0, sizeof(j));
		f = 0;
		r = 0;
	}

	__syncthreads();

	if (!blockHasValidPoint)
		return;

	parallelReduce<256>(accumulator[blockId].numPoints, (int) isValidPoint, threadId);
	parallelReduce<256>(accumulator[blockId].f, f, threadId);
	parallelReduceArray4<256>(accumulator[blockId].g.v, (j * r).v, threadId);

	float localHessian[4 * 4];
#if (defined(__CUDACC__) && defined(__CUDA_ARCH__))
#pragma unroll
#endif
	for (unsigned char r = 0, counter = 0; r < 4; r++)
	{
#if (defined(__CUDACC__) && defined(__CUDA_ARCH__))
#pragma unroll
#endif
		for (int c = 0; c < 4; c++, counter++) localHessian[counter] = j[r] * j[c];
	}

	for (int offset = 0; offset < 4 * 4; offset = offset + 4)
		parallelReduceArray4<256>(accumulator[blockId].H.m + offset, localHessian + offset, threadId);
}

template<bool shortIteration, bool rotationOnly>
__global__ void
depthTrackerOneLevel_g_rt_device(KernelParameters para)
{
	depthTrackerOneLevel_g_rt_device_main<shortIteration, rotationOnly>(para.accu, para.depth,
	                                                                    para.approxInvPose,
	                                                                    para.pointsMap, para.normalsMap,
	                                                                    para.sceneIntrinsics, para.sceneImageSize,
	                                                                    para.scenePose, para.viewIntrinsics,
	                                                                    para.viewImageSize, para.scaleFactor,
	                                                                    para.distThresh);
}

template<bool shortIteration, bool rotationOnly>
__global__ void RGBTrackerOneLevel_g_rt_device(RGBKernelParameters para)
{
	RGBTrackerOneLevel_g_rt_device_main<shortIteration, rotationOnly>(para.accu,
	                                                                  para.points_current,
	                                                                  para.intensity_current,
	                                                                  para.intensity_reference,
	                                                                  para.gradient_reference,
	                                                                  para.approxInvPose,
	                                                                  para.intensityReferencePose,
	                                                                  para.intrinsics_depth,
	                                                                  para.intrinsics_rgb,
	                                                                  para.imgSize_depth,
	                                                                  para.imgSize_rgb,
	                                                                  para.viewFrustum_max,
	                                                                  para.intensityThresh,
	                                                                  para.minGradient);
}