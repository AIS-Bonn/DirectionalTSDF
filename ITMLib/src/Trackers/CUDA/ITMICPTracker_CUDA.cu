// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMICPTracker_CUDA.h"
#include "../Shared/ITMICPTracker_Shared.h"
#include <Trackers/Shared/ITMTracker_Shared.h>
#include <Utils/ITMCUDAUtils.h>
#include <ORUtils/CUDADefines.h>

using namespace ITMLib;

struct ITMICPTracker_CUDA::AccuCell
{
	int numPoints;
	float f;
	float g[6];
	float h[6 + 5 + 4 + 3 + 2 + 1];
};

struct ITMICPTracker_KernelParameters
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
	float distThresh;
};

struct ITMICPTracker_RGBKernelParameters
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

template<bool shortIteration, bool rotationOnly>
__global__ void depthTrackerOneLevel_g_rt_device(ITMICPTracker_KernelParameters para);

template<bool shortIteration, bool rotationOnly>
__global__ void RGBTrackerOneLevel_g_rt_device(ITMICPTracker_RGBKernelParameters para);

// host methods

ITMICPTracker_CUDA::ITMICPTracker_CUDA(Vector2i imgSize_d, Vector2i imgSize_rgb, const Parameters& parameters,
                                       const ITMLowLevelEngine* lowLevelEngine)
	: ITMICPTracker(imgSize_d, imgSize_rgb, parameters, lowLevelEngine, MEMORYDEVICE_CUDA)
{
	ORcudaSafeCall(cudaMallocHost((void**) &accu_host, sizeof(AccuCell)));
	ORcudaSafeCall(cudaMalloc((void**) &accu_device, sizeof(AccuCell)));
}

ITMICPTracker_CUDA::~ITMICPTracker_CUDA(void)
{
	ORcudaSafeCall(cudaFreeHost(accu_host));
	ORcudaSafeCall(cudaFree(accu_device));
}

int ITMICPTracker_CUDA::ComputeGandH_Depth(float& f, float* nabla, float* hessian, Matrix4f approxInvPose)
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

	ORcudaSafeCall(cudaMemset(accu_device, 0, sizeof(AccuCell)));

	struct ITMICPTracker_KernelParameters args;
	args.accu = accu_device;
	args.depth = depth;
	args.approxInvPose = approxInvPose;
	args.pointsMap = pointsMap;
	args.normalsMap = normalsMap;
	args.sceneIntrinsics = sceneIntrinsics;
	args.sceneImageSize = sceneImageSize;
	args.scenePose = renderedScenePose;
	args.viewIntrinsics = viewIntrinsics;
	args.viewImageSize = viewImageSize;
	args.distThresh = distThresh[levelId];

	switch (iterationType)
	{
		case TRACKER_ITERATION_ROTATION:
			depthTrackerOneLevel_g_rt_device<true, true> << < gridSize, blockSize >> >(args);
			ORcudaKernelCheck;
			break;
		case TRACKER_ITERATION_TRANSLATION:
			depthTrackerOneLevel_g_rt_device<true, false> << < gridSize, blockSize >> >(args);
			ORcudaKernelCheck;
			break;
		case TRACKER_ITERATION_BOTH:
			depthTrackerOneLevel_g_rt_device<false, false> << < gridSize, blockSize >> >(args);
			ORcudaKernelCheck;
			break;
		default:
			break;
	}

	ORcudaSafeCall(cudaMemcpy(accu_host, accu_device, sizeof(AccuCell), cudaMemcpyDeviceToHost));

	if (accu_host->numPoints == 0)
		return 0;

	float invNoPoints = 1.0f / accu_host->numPoints;

	for (int r = 0, counter = 0; r < noPara; r++)
		for (int c = 0; c <= r; c++, counter++)
			hessian[r + c * 6] = accu_host->h[counter] * invNoPoints;
	for (int r = 0; r < noPara; ++r)
		for (int c = r + 1; c < noPara; c++)
			hessian[r + c * 6] = hessian[c + r * 6];

	for (int c = 0; c < noPara; c++)
		nabla[c] = accu_host->g[c] * invNoPoints;

	f = accu_host->f * invNoPoints;

	return accu_host->numPoints;
}


int ITMICPTracker_CUDA::ComputeGandH_RGB(float& f, float* nabla, float* hessian, Matrix4f approxInvPose)
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

	ORcudaSafeCall(cudaMemset(accu_device, 0, sizeof(AccuCell)));

	struct ITMICPTracker_RGBKernelParameters args;
	args.accu = accu_device;
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
	args.viewFrustum_max = 6;
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

	ORcudaSafeCall(cudaMemcpy(accu_host, accu_device, sizeof(AccuCell), cudaMemcpyDeviceToHost));

	if (accu_host->numPoints == 0)
		return 0;

	float invNoPoints = 1.0f / accu_host->numPoints;

	for (int r = 0, counter = 0; r < noPara; r++)
		for (int c = 0; c <= r; c++, counter++)
			hessian[r + c * 6] = accu_host->h[counter] * invNoPoints;
	for (int r = 0; r < noPara; ++r)
		for (int c = r + 1; c < noPara; c++)
			hessian[r + c * 6] = hessian[c + r * 6];

	for (int c = 0; c < noPara; c++)
		nabla[c] = accu_host->g[c] * invNoPoints;

	f = accu_host->f * invNoPoints;

//	printf("f = %f\n", f);
//	printf("n = (");
//	for (int c = 0; c < noPara; c++)
//		printf("%.1f\t", nabla[c]);
//	printf(")\n");
//	printf("JJ =");
//	for (int r = 0; r < noPara; r++)
//	{
//		printf("\t");
//		if (r > 0)
//			printf("\t");
//		for (int c = 0; c < noPara; c++)
//			printf("%14.1f\t", hessian[r + c * 6]);
//		printf("\n");
//	}
//	printf("\n");

	return accu_host->numPoints;
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
RGBTrackerOneLevel_g_rt_device_main(ITMICPTracker_RGBKernelParameters para)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	int locId_local = threadIdx.x + threadIdx.y * blockDim.x;

	__shared__ float dim_shared1[256];
	__shared__ float dim_shared2[256];
	__shared__ float dim_shared3[256];
	__shared__ bool should_prefix;

	should_prefix = false;
	__syncthreads();

	const int noPara = shortIteration ? 3 : 6;
	const int noParaSQ = shortIteration ? 3 + 2 + 1 : 6 + 5 + 4 + 3 + 2 + 1;
	float A[noPara];
	float b;
	bool isValidPoint = false;

	for (int i = 0; i < noPara; i++) A[i] = 0.0f;

	if (x < para.imgSize_depth.x && y < para.imgSize_depth.y)
	{
		isValidPoint = computePerPointGH_RGB_Ab<shortIteration, rotationOnly>(A, b, x, y,
		                                                                      para.points_current,
		                                                                      para.intensity_current,
		                                                                      para.intensity_reference,
		                                                                      para.gradient_reference,
		                                                                      para.imgSize_depth,
		                                                                      para.imgSize_rgb,
		                                                                      para.intrinsics_depth,
		                                                                      para.intrinsics_rgb,
		                                                                      para.approxInvPose,
		                                                                      para.intensityReferencePose,
		                                                                      para.viewFrustum_max,
		                                                                      para.intensityThresh,
		                                                                      para.minGradient
		);
		if (isValidPoint) should_prefix = true;
	}

	if (!isValidPoint)
	{
		for (int i = 0; i < noPara; i++) A[i] = 0.0f;
		b = 0.0f;
	}

	__syncthreads();

	if (!should_prefix) return;

	{ //reduction for noValidPoints
		dim_shared1[locId_local] = isValidPoint;
		__syncthreads();

		if (locId_local < 128) dim_shared1[locId_local] += dim_shared1[locId_local + 128];
		__syncthreads();
		if (locId_local < 64) dim_shared1[locId_local] += dim_shared1[locId_local + 64];
		__syncthreads();

		if (locId_local < 32) warpReduce(dim_shared1, locId_local);

		if (locId_local == 0) atomicAdd(&(para.accu->numPoints), (int) dim_shared1[locId_local]);
	}

	{ //reduction for energy function value
		dim_shared1[locId_local] = b * b;
		__syncthreads();

		if (locId_local < 128) dim_shared1[locId_local] += dim_shared1[locId_local + 128];
		__syncthreads();
		if (locId_local < 64) dim_shared1[locId_local] += dim_shared1[locId_local + 64];
		__syncthreads();

		if (locId_local < 32) warpReduce(dim_shared1, locId_local);

		if (locId_local == 0) atomicAdd(&(para.accu->f), dim_shared1[locId_local]);
	}

	__syncthreads();

	//reduction for nabla
	for (unsigned char paraId = 0; paraId < noPara; paraId += 3)
	{
		dim_shared1[locId_local] = b * A[paraId + 0];
		dim_shared2[locId_local] = b * A[paraId + 1];
		dim_shared3[locId_local] = b * A[paraId + 2];
		__syncthreads();

		if (locId_local < 128)
		{
			dim_shared1[locId_local] += dim_shared1[locId_local + 128];
			dim_shared2[locId_local] += dim_shared2[locId_local + 128];
			dim_shared3[locId_local] += dim_shared3[locId_local + 128];
		}
		__syncthreads();
		if (locId_local < 64)
		{
			dim_shared1[locId_local] += dim_shared1[locId_local + 64];
			dim_shared2[locId_local] += dim_shared2[locId_local + 64];
			dim_shared3[locId_local] += dim_shared3[locId_local + 64];
		}
		__syncthreads();

		if (locId_local < 32)
		{
			warpReduce(dim_shared1, locId_local);
			warpReduce(dim_shared2, locId_local);
			warpReduce(dim_shared3, locId_local);
		}
		__syncthreads();

		if (locId_local == 0)
		{
			atomicAdd(&(para.accu->g[paraId + 0]), dim_shared1[0]);
			atomicAdd(&(para.accu->g[paraId + 1]), dim_shared2[0]);
			atomicAdd(&(para.accu->g[paraId + 2]), dim_shared3[0]);
		}
	}

	__syncthreads();

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
		dim_shared1[locId_local] = localHessian[paraId + 0];
		dim_shared2[locId_local] = localHessian[paraId + 1];
		dim_shared3[locId_local] = localHessian[paraId + 2];
		__syncthreads();

		if (locId_local < 128)
		{
			dim_shared1[locId_local] += dim_shared1[locId_local + 128];
			dim_shared2[locId_local] += dim_shared2[locId_local + 128];
			dim_shared3[locId_local] += dim_shared3[locId_local + 128];
		}
		__syncthreads();
		if (locId_local < 64)
		{
			dim_shared1[locId_local] += dim_shared1[locId_local + 64];
			dim_shared2[locId_local] += dim_shared2[locId_local + 64];
			dim_shared3[locId_local] += dim_shared3[locId_local + 64];
		}
		__syncthreads();

		if (locId_local < 32)
		{
			warpReduce(dim_shared1, locId_local);
			warpReduce(dim_shared2, locId_local);
			warpReduce(dim_shared3, locId_local);
		}
		__syncthreads();

		if (locId_local == 0)
		{
			atomicAdd(&(para.accu->h[paraId + 0]), dim_shared1[0]);
			atomicAdd(&(para.accu->h[paraId + 1]), dim_shared2[0]);
			atomicAdd(&(para.accu->h[paraId + 2]), dim_shared3[0]);
		}
	}
}

template<bool shortIteration, bool rotationOnly>
__device__ void
depthTrackerOneLevel_g_rt_device_main(ITMICPTracker_CUDA::AccuCell* accu, float* depth, Matrix4f approxInvPose,
                                      Vector4f* pointsMap,
                                      Vector4f* normalsMap, Vector4f sceneIntrinsics, Vector2i sceneImageSize,
                                      Matrix4f scenePose, Vector4f viewIntrinsics, Vector2i viewImageSize,
                                      float distThresh)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;

	int locId_local = threadIdx.x + threadIdx.y * blockDim.x;

	__shared__ float dim_shared1[256];
	__shared__ float dim_shared2[256];
	__shared__ float dim_shared3[256];
	__shared__ bool should_prefix;

	should_prefix = false;
	__syncthreads();

	const int noPara = shortIteration ? 3 : 6;
	const int noParaSQ = shortIteration ? 3 + 2 + 1 : 6 + 5 + 4 + 3 + 2 + 1;
	float A[noPara];
	float b;
	bool isValidPoint = false;

	if (x < viewImageSize.x && y < viewImageSize.y)
	{
		isValidPoint = computePerPointGH_Depth_Ab<shortIteration, rotationOnly>(A, b, x, y, depth[x + y * viewImageSize.x],
		                                                                        viewImageSize, viewIntrinsics,
		                                                                        sceneImageSize, sceneIntrinsics,
		                                                                        approxInvPose, scenePose, pointsMap,
		                                                                        normalsMap, distThresh);
		if (isValidPoint) should_prefix = true;
	}

	if (!isValidPoint)
	{
		for (int i = 0; i < noPara; i++) A[i] = 0.0f;
		b = 0.0f;
	}

	__syncthreads();

	if (!should_prefix) return;

	{ //reduction for noValidPoints
		dim_shared1[locId_local] = isValidPoint;
		__syncthreads();

		if (locId_local < 128) dim_shared1[locId_local] += dim_shared1[locId_local + 128];
		__syncthreads();
		if (locId_local < 64) dim_shared1[locId_local] += dim_shared1[locId_local + 64];
		__syncthreads();

		if (locId_local < 32) warpReduce(dim_shared1, locId_local);

		if (locId_local == 0) atomicAdd(&(accu->numPoints), (int) dim_shared1[locId_local]);
	}

	{ //reduction for energy function value
		dim_shared1[locId_local] = b * b;
		__syncthreads();

		if (locId_local < 128) dim_shared1[locId_local] += dim_shared1[locId_local + 128];
		__syncthreads();
		if (locId_local < 64) dim_shared1[locId_local] += dim_shared1[locId_local + 64];
		__syncthreads();

		if (locId_local < 32) warpReduce(dim_shared1, locId_local);

		if (locId_local == 0) atomicAdd(&(accu->f), dim_shared1[locId_local]);
	}

	__syncthreads();

	//reduction for nabla
	for (unsigned char paraId = 0; paraId < noPara; paraId += 3)
	{
		dim_shared1[locId_local] = b * A[paraId + 0];
		dim_shared2[locId_local] = b * A[paraId + 1];
		dim_shared3[locId_local] = b * A[paraId + 2];
		__syncthreads();

		if (locId_local < 128)
		{
			dim_shared1[locId_local] += dim_shared1[locId_local + 128];
			dim_shared2[locId_local] += dim_shared2[locId_local + 128];
			dim_shared3[locId_local] += dim_shared3[locId_local + 128];
		}
		__syncthreads();
		if (locId_local < 64)
		{
			dim_shared1[locId_local] += dim_shared1[locId_local + 64];
			dim_shared2[locId_local] += dim_shared2[locId_local + 64];
			dim_shared3[locId_local] += dim_shared3[locId_local + 64];
		}
		__syncthreads();

		if (locId_local < 32)
		{
			warpReduce(dim_shared1, locId_local);
			warpReduce(dim_shared2, locId_local);
			warpReduce(dim_shared3, locId_local);
		}
		__syncthreads();

		if (locId_local == 0)
		{
			atomicAdd(&(accu->g[paraId + 0]), dim_shared1[0]);
			atomicAdd(&(accu->g[paraId + 1]), dim_shared2[0]);
			atomicAdd(&(accu->g[paraId + 2]), dim_shared3[0]);
		}
	}

	__syncthreads();

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
		dim_shared1[locId_local] = localHessian[paraId + 0];
		dim_shared2[locId_local] = localHessian[paraId + 1];
		dim_shared3[locId_local] = localHessian[paraId + 2];
		__syncthreads();

		if (locId_local < 128)
		{
			dim_shared1[locId_local] += dim_shared1[locId_local + 128];
			dim_shared2[locId_local] += dim_shared2[locId_local + 128];
			dim_shared3[locId_local] += dim_shared3[locId_local + 128];
		}
		__syncthreads();
		if (locId_local < 64)
		{
			dim_shared1[locId_local] += dim_shared1[locId_local + 64];
			dim_shared2[locId_local] += dim_shared2[locId_local + 64];
			dim_shared3[locId_local] += dim_shared3[locId_local + 64];
		}
		__syncthreads();

		if (locId_local < 32)
		{
			warpReduce(dim_shared1, locId_local);
			warpReduce(dim_shared2, locId_local);
			warpReduce(dim_shared3, locId_local);
		}
		__syncthreads();

		if (locId_local == 0)
		{
			atomicAdd(&(accu->h[paraId + 0]), dim_shared1[0]);
			atomicAdd(&(accu->h[paraId + 1]), dim_shared2[0]);
			atomicAdd(&(accu->h[paraId + 2]), dim_shared3[0]);
		}
	}
}

template<bool shortIteration, bool rotationOnly>
__global__ void depthTrackerOneLevel_g_rt_device(ITMICPTracker_KernelParameters para)
{
	depthTrackerOneLevel_g_rt_device_main<shortIteration, rotationOnly>(para.accu, para.depth, para.approxInvPose,
	                                                                    para.pointsMap, para.normalsMap,
	                                                                    para.sceneIntrinsics, para.sceneImageSize,
	                                                                    para.scenePose, para.viewIntrinsics,
	                                                                    para.viewImageSize, para.distThresh);
}

template<bool shortIteration, bool rotationOnly>
__global__ void RGBTrackerOneLevel_g_rt_device(ITMICPTracker_RGBKernelParameters para)
{
	RGBTrackerOneLevel_g_rt_device_main<shortIteration, rotationOnly>(para);
}