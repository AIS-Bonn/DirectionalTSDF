// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../../Objects/Tracking/ITMTrackingState.h"
#include "../../../Objects/Views/ITMView.h"

#include "../Shared/ITMVisualisationEngine_Shared.h"
#include "../../../Utils/ITMCUDAUtils.h"
#include <ITMLib/Objects/Scene/TSDF_CUDA.h>

#include <stdgpu/unordered_set_fwd>
#include <stdgpu/unordered_map.cuh>

namespace ITMLib
{
// declaration of device functions

__global__ void
projectAndSplitBlocks_device(RenderingBlock* renderingBlocks, uint* noTotalBlocks, const ITMIndex* visibleBlocks,
                             int noVisibleEntries, const Matrix4f pose_M, const Vector4f intrinsics,
                             const Vector2i imgSize, float voxelSize);

__global__ void computeMinMaxData_device(uint noTotalBlocks, const RenderingBlock* renderingBlocks,
                                         Vector2i imgSize, Vector2f* minmaxData);

__global__ void findMissingPoints_device(int* fwdProjMissingPoints, uint* noMissingPoints, const Vector2f* minmaximg,
                                         Vector4f* forwardProjection, float* currentDepth, Vector2i imgSize);

__global__ void
forwardProject_device(Vector4f* forwardProjection, const Vector4f* pointsRay, Vector2i imgSize, Matrix4f M,
                      Vector4f projParams, float voxelSize);

__global__ void
renderICP_device(Vector4f* pointsMap, Vector4f* normalsMap, const Vector4f* pointsRay, const Vector4f* normalsRay,
                 float voxelSize, Vector2i imgSize, Vector3f lightSource);

__global__ void
renderDepthShaded_ImageNormals_device(Vector4u* outRendering, const Vector4f* pointsRay, const Vector4f* normalsRay,
                                      Vector2i imgSize, Vector3f lightSource);

__global__ void
renderNormals_ImageNormals_device(Vector4u* outRendering, const Vector4f* ptsRay, const Vector4f* normalsRay,
                                  Vector2i imgSize, float voxelSize, Vector3f lightSource);

__global__ void
renderConfidence_ImageNormals_device(Vector4u* outRendering, const Vector4f* ptsRay,
                                     const Vector4f* normalsRay, Vector2i imgSize,
                                     const ITMSceneParams sceneParams, Vector3f lightSource);

/**
 * A faster version of combineDiretionalTSDFViewPoint that uses shared memory between threads to reduce tsdf lookups
 * @param renderingTSDF
 * @param tsdf
 * @param visibleBlocks
 * @param numVisibleBlocks
 * @param invM
 * @param voxelSize
 * @param mu
 * @param maxW
 */
__global__ void
combineDirectionalTSDFViewPoint_opt_device(stdgpu::unordered_map<ITMIndex, ITMVoxel*> renderingTSDF,
                                           const stdgpu::unordered_map<ITMIndexDirectional, ITMVoxel*> tsdf,
                                           const ITMIndex* visibleBlocks, const stdgpu::index_t numVisibleBlocks,
                                           const Matrix4f invM, const float voxelSize, const float mu, const int maxW);

__global__ void
combineDirectionalTSDFViewPoint_device(stdgpu::unordered_map<ITMIndex, ITMVoxel*> renderingTSDF,
                                       const stdgpu::unordered_map<ITMIndexDirectional, ITMVoxel*> tsdf,
                                       const ITMIndex* visibleBlocks, const stdgpu::index_t numVisibleBlocks,
                                       const Matrix4f invM, const float voxelSize, const float mu, const int maxW);

template<class TIndex, class TVoxel>
__global__ void genericRaycast_device(Vector4f* out_ptsRay,
                                      const stdgpu::unordered_map<TIndex, TVoxel*> tsdf, Vector2i imgSize,
                                      Matrix4f invM, Vector4f invProjParams,
                                      const ITMSceneParams sceneParams, const Vector2f* minmaximg,
                                      TSDFDirection direction = TSDFDirection::NONE)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	int locId = x + y * imgSize.x;
	int locId2 = (int) floor((float) x / minmaximg_subsample) + (int) floor((float) y / minmaximg_subsample) * imgSize.x;

	float distance;
	castRayDefaultTSDF(out_ptsRay[locId], distance, x, y,
	                   tsdf, invM, invProjParams, sceneParams, minmaximg[locId2], direction);
}

template<class TIndex, class TVoxel, template<typename, typename...> class Map, typename... Args>
__global__ void renderDepthShaded_device(Vector4u* outRendering, const Vector4f* ptsRay,
                                         const Map<TIndex, TVoxel*, Args...> tsdf,
                                         float oneOverVoxelSize, Vector2i imgSize, Vector3f lightSource)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	int locId = x + y * imgSize.x;

	processPixelDepthShaded_SDFNormals(outRendering[locId], ptsRay[locId], tsdf, oneOverVoxelSize, lightSource);
}

template<class TIndex, class TVoxel, template<typename, typename...> class Map, typename... Args>
__global__ void renderNormals_device(Vector4u* outRendering, const Vector4f* ptsRay,
                                     const Map<TIndex, TVoxel*, Args...> tsdf,
                                     const Vector2i imgSize,
                                     const float oneOverVoxelSize,
                                     const Vector3f lightSource)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	int locId = x + y * imgSize.x;

	processPixelNormal_SDFNormals(outRendering[locId], ptsRay[locId], tsdf, oneOverVoxelSize, lightSource);
}

template<class TIndex, class TVoxel, template<typename, typename...> class Map, typename... Args>
__global__ void renderConfidence_device(Vector4u* outRendering, const Vector4f* ptsRay,
                                        const Map<TIndex, TVoxel*, Args...> tsdf, Vector2i imgSize,
                                        const ITMSceneParams sceneParams, const Vector3f lightSource)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	int locId = x + y * imgSize.x;

	processPixelConfidence_SDFNormals(outRendering[locId], ptsRay[locId], tsdf, sceneParams, lightSource);
}

template<class TIndex, class TVoxel, template<typename, typename...> class Map, typename... Args>
__global__ void renderPointCloud_device(Vector4f* locations, Vector4f* colours, uint* noTotalPoints,
                                        const Vector4f* ptsRay, const Map<TIndex, TVoxel*, Args...> tsdf,
                                        const bool skipPoints,
                                        float voxelSize, const Vector2i imgSize, const Vector3f lightSource)
{
	__shared__ bool shouldPrefix;
	shouldPrefix = false;
	__syncthreads();


	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x && y >= imgSize.y)
		return;

	int locId = x + y * imgSize.x;
	Vector3f outNormal;
	float angle;

	const Vector4f& pointRay = ptsRay[locId];
	Vector3f point = pointRay.toVector3();
	bool foundPoint = pointRay.w > 0;
	computeNormalAndAngleTSDF(foundPoint, point, tsdf, 1 / voxelSize, lightSource, outNormal, angle);

	if (skipPoints && ((x % 2 == 0) || (y % 2 == 0))) foundPoint = false;

	if (foundPoint) shouldPrefix = true;

	__syncthreads();

	if (shouldPrefix)
	{
		int offset = computePrefixSum_device<uint>(foundPoint, noTotalPoints, blockDim.x * blockDim.y,
		                                           threadIdx.x + threadIdx.y * blockDim.x);

		if (offset != -1)
		{
			colours[offset] = readFromSDF_color4u_interpolated(tsdf, point);
			locations[offset] = Vector4f(point, 1.0);
		}
	}
}

template<class TIndex, class TVoxel, template<typename, typename...> class Map, typename... Args>
__global__ void renderColour_device(Vector4u* outRendering, const Vector4f* ptsRay,
                                    const Map<TIndex, TVoxel*, Args...> tsdf, const float oneOverVoxelSize,
                                    Vector2i imgSize, const Vector3f lightSource)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	int locId = x + y * imgSize.x;

	processPixelColour(outRendering[locId], ptsRay[locId], tsdf, oneOverVoxelSize, lightSource);
}

template<class TVoxel>
__global__ void renderDepthColour_device(Vector4u* outRendering, const Vector4f* ptsRay, const Matrix4f T_CW,
                                         const Vector2i imgSize, const float maxDepth)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	int locId = x + y * imgSize.x;

	processPixelDepthColour<TVoxel>(outRendering[locId], ptsRay[locId], T_CW, maxDepth);
}

template<class TVoxel>
__global__ void computePointCloudNormals_device(Vector4f* outputNormals, const Vector4f* pointsRay,
                                                const Vector2i imgSize, float voxelSize)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x);
	int y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	bool foundPoint = true;
	Vector3f normal;
	computeNormal<false>(pointsRay, voxelSize, imgSize, x, y, foundPoint, normal);

	if (not foundPoint)
	{
		outputNormals[x + y * imgSize.x] = Vector4f(0, 0, 0, -1);
		return;
	}

	outputNormals[x + y * imgSize.x] = Vector4f(normal, 1);
}


template<class TVoxel>
__global__ void renderPixelError_device(
	Vector4u* outRendering, const Vector4f* pointsRay, const Vector4f* normalsRay, const float* depth,
	const Matrix4f depthImagePose, const Matrix4f sceneRenderingPose, const Vector4f intrinsics,
	const Vector2i imgSize, const float maxError)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x);
	int y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	processPixelError(outRendering, pointsRay, normalsRay, depth, depthImagePose, sceneRenderingPose, intrinsics,
	                  imgSize, maxError, x, y);
}

} // ITMLib
