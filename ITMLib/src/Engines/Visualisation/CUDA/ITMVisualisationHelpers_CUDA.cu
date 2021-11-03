// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMVisualisationHelpers_CUDA.h"
#include <Engines/Reconstruction/Shared/ITMSceneReconstructionEngine_Shared.h>
#include <ITMLib/Engines/ITMSceneReconstructionEngine.h>

#include <cstddef>
#include <stdgpu/cstddef.h>
#include <stdgpu/unordered_map.cuh>
#include <stdgpu/unordered_set.cuh>
#include <thrust/copy.h>

//device implementations

namespace ITMLib
{

__global__ void
projectAndSplitBlocks_device(RenderingBlock* renderingBlocks, uint* noTotalBlocks, const ITMIndex* visibleBlocks,
                             int noVisibleEntries, const Matrix4f pose_M, const Vector4f intrinsics,
                             const Vector2i imgSize, float voxelSize)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	Vector2i upperLeft, lowerRight;
	Vector2f zRange;
	bool validProjection = false;
	if (idx < noVisibleEntries)
	{
		Vector3s blockIdx = visibleBlocks[idx].getPosition().toShort();
		validProjection = ProjectSingleBlock(blockIdx, pose_M, intrinsics, imgSize, voxelSize, upperLeft, lowerRight,
		                                     zRange);
	}

	Vector2i requiredRenderingBlocks(ceilf((float) (lowerRight.x - upperLeft.x + 1) / renderingBlockSizeX),
	                                 ceilf((float) (lowerRight.y - upperLeft.y + 1) / renderingBlockSizeY));

	size_t requiredNumBlocks = requiredRenderingBlocks.x * requiredRenderingBlocks.y;
	if (!validProjection) requiredNumBlocks = 0;

	int out_offset = computePrefixSum_device<uint>(requiredNumBlocks, noTotalBlocks, blockDim.x, threadIdx.x);
	if (!validProjection) return;
	if ((out_offset == -1) || (out_offset + requiredNumBlocks > MAX_RENDERING_BLOCKS)) return;

	CreateRenderingBlocks(renderingBlocks, out_offset, upperLeft, lowerRight, zRange);
}

__global__ void computeMinMaxData_device(uint noTotalBlocks, const RenderingBlock* renderingBlocks,
                                         Vector2i imgSize, Vector2f* minmaxData)
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	int block = blockIdx.x * 4 + blockIdx.y;
	if (block >= noTotalBlocks) return;

	const RenderingBlock& b(renderingBlocks[block]);
	int xpos = b.upperLeft.x + x;
	if (xpos > b.lowerRight.x) return;
	int ypos = b.upperLeft.y + y;
	if (ypos > b.lowerRight.y) return;

	Vector2f& pixel(minmaxData[xpos + ypos * imgSize.x]);
	atomicMin(&pixel.x, b.zRange.x);
	atomicMax(&pixel.y, b.zRange.y);
}

__global__ void findMissingPoints_device(int* fwdProjMissingPoints, uint* noMissingPoints, const Vector2f* minmaximg,
                                         Vector4f* forwardProjection, float* currentDepth, Vector2i imgSize)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	int locId = x + y * imgSize.x;
	int locId2 = (int) floor((float) x / minmaximg_subsample) + (int) floor((float) y / minmaximg_subsample) * imgSize.x;

	Vector4f fwdPoint = forwardProjection[locId];
	Vector2f minmaxval = minmaximg[locId2];
	float depth = currentDepth[locId];

	bool hasPoint = false;

	__shared__ bool shouldPrefix;
	shouldPrefix = false;
	__syncthreads();

	if ((fwdPoint.w <= 0) && ((fwdPoint.x == 0 && fwdPoint.y == 0 && fwdPoint.z == 0) || (depth > 0)) &&
	    (minmaxval.x < minmaxval.y))
		//if ((fwdPoint.w <= 0) && (minmaxval.x < minmaxval.y))
	{
		shouldPrefix = true;
		hasPoint = true;
	}

	__syncthreads();

	if (shouldPrefix)
	{
		int offset = computePrefixSum_device(hasPoint, noMissingPoints, blockDim.x * blockDim.y,
		                                     threadIdx.x + threadIdx.y * blockDim.x);
		if (offset != -1) fwdProjMissingPoints[offset] = locId;
	}
}

__global__ void
forwardProject_device(Vector4f* forwardProjection, const Vector4f* pointsRay, Vector2i imgSize, Matrix4f M,
                      Vector4f projParams, float voxelSize)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	int locId = x + y * imgSize.x;
	Vector4f pixel = pointsRay[locId];

	int locId_new = forwardProjectPixel(pixel * voxelSize, M, projParams, imgSize);
	if (locId_new >= 0) forwardProjection[locId_new] = pixel;
}

__global__ void
renderDepthShaded_ImageNormals_device(Vector4u* outRendering, const Vector4f* pointsRay, const Vector4f* normalsRay,
                                      Vector2i imgSize, Vector3f lightSource)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	processPixelDepthShaded_ImageNormals<true>(outRendering, pointsRay, normalsRay, imgSize, x, y, lightSource);
}

__global__ void
renderNormals_ImageNormals_device(Vector4u* outRendering, const Vector4f* ptsRay, const Vector4f* normalsRay,
                                  Vector2i imgSize, float voxelSize, Vector3f lightSource)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	processPixelNormals_ImageNormals<true>(outRendering, ptsRay, normalsRay, imgSize, x, y, voxelSize, lightSource);
}

__global__ void
renderConfidence_ImageNormals_device(Vector4u* outRendering, const Vector4f* ptsRay,
                                     const Vector4f* normalsRay, Vector2i imgSize,
                                     const ITMSceneParams sceneParams, Vector3f lightSource)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x), y = (threadIdx.y + blockIdx.y * blockDim.y);

	if (x >= imgSize.x || y >= imgSize.y) return;

	processPixelConfidence_ImageNormals<true>(outRendering, ptsRay, normalsRay, imgSize, x, y, sceneParams, lightSource);
}

#define ReadVoxelSDFDiscardZero(dst, blk, pos) voxel = &blk[VoxelIndicesToOffset(pos.x, pos.y, pos.z)]; if (voxel->w_depth == 0) return Vector3f(0, 0, 0); dst = voxel->sdf;

_CPU_AND_GPU_CODE_ inline Vector3f
computeGradient(ITMVoxel* block, ITMVoxel** neighbors, const Vector3i& idx, float tau)
{
	if (not block)
		return Vector3f(0, 0, 0);

	if (idx.x == 0 and not neighbors[1])
		return Vector3f(0, 0, 0);
	if (idx.x == 7 and not neighbors[0])
		return Vector3f(0, 0, 0);
	if (idx.y == 0 and not neighbors[3])
		return Vector3f(0, 0, 0);
	if (idx.y == 7 and not neighbors[2])
		return Vector3f(0, 0, 0);
	if (idx.z == 0 and not neighbors[5])
		return Vector3f(0, 0, 0);
	if (idx.z == 7 and not neighbors[4])
		return Vector3f(0, 0, 0);

	ITMVoxel* voxel;
	Vector3f ret(0, 0, 0);
	float p1, p2;

	// gradient x
	if (idx.x == 7)
	{
		ReadVoxelSDFDiscardZero(p1, neighbors[static_cast<TSDFDirection_type>(TSDFDirection::X_POS)],
		                        Vector3i(0, idx.y, idx.z));
	} else
	{
		ReadVoxelSDFDiscardZero(p1, block, Vector3i(idx.x + 1, idx.y, idx.z));
	}
	if (idx.x == 0)
	{
		ReadVoxelSDFDiscardZero(p2, neighbors[1], Vector3i(7, idx.y, idx.z));
	} else
	{
		ReadVoxelSDFDiscardZero(p2, block, Vector3i(idx.x - 1, idx.y, idx.z));
	}
	ret.x = ITMVoxel::valueToFloat(p1 - p2);

	// gradient y
	if (idx.y == 7)
	{
		ReadVoxelSDFDiscardZero(p1, neighbors[2], Vector3i(idx.x, 0, idx.z));
	} else
	{
		ReadVoxelSDFDiscardZero(p1, block, Vector3i(idx.x, idx.y + 1, idx.z));
	}
	if (idx.y == 0)
	{
		ReadVoxelSDFDiscardZero(p2, neighbors[3], Vector3i(idx.x, 7, idx.z));
	} else
	{
		ReadVoxelSDFDiscardZero(p2, block, Vector3i(idx.x, idx.y - 1, idx.z));
	}
	ret.y = ITMVoxel::valueToFloat(p1 - p2);

	// gradient z
	if (idx.z == 7)
	{
		ReadVoxelSDFDiscardZero(p1, neighbors[4], Vector3i(idx.x, idx.y, 0));
	} else
	{
		ReadVoxelSDFDiscardZero(p1, block, Vector3i(idx.x, idx.y, idx.z + 1));
	}
	if (idx.z == 0)
	{
		ReadVoxelSDFDiscardZero(p2, neighbors[5], Vector3i(idx.x, idx.y, 7));
	} else
	{
		ReadVoxelSDFDiscardZero(p2, block, Vector3i(idx.x, idx.y, idx.z - 1));
	}
	ret.z = ITMVoxel::valueToFloat(p1 - p2);

	if (ret == Vector3f(0, 0, 0))
		return Vector3f(0, 0, 0);

	if (tau > 0)
	{
		// Check each direction maximum 2 * truncationDistance / voxelSize + margin
		if (abs(ret.x) > 2.4 * tau or abs(ret.y) > 2.4 * tau or abs(ret.z) > 2.4 * tau)
			return Vector3f(0, 0, 0);
		// Check, if ret too unreliable (very close values in neighboring voxels). minimum expected length: (2 * tau)^2
		if (ORUtils::dot(ret, ret) < 2 * (tau * tau)) return Vector3f(0, 0, 0);
	}

	return ret.normalised();
}

__global__ void
combineDirectionalTSDFViewPoint_opt_device(stdgpu::unordered_map<ITMIndex, ITMVoxel*> renderingTSDF,
                                           const stdgpu::unordered_map<ITMIndexDirectional, ITMVoxel*> tsdf,
                                           const ITMIndex* visibleBlocks, const stdgpu::index_t numVisibleBlocks,
                                           const Matrix4f invM, const float voxelSize, const float mu, const int maxW)
{
	stdgpu::index_t i = static_cast<stdgpu::index_t>(blockIdx.x);
	if (i >= numVisibleBlocks) return;

	int linearIdx = blockIdx.y * blockDim.x + threadIdx.x;

	const ITMIndex& blockPos = visibleBlocks[i];

	__shared__ ITMVoxel* renderBlock;
	__shared__ ITMVoxel* blocks[N_DIRECTIONS];
	__shared__ ITMVoxel* neighborBlocks[6 * N_DIRECTIONS];
	if (linearIdx == 6)
	{
		auto it = renderingTSDF.find(blockPos);
		renderBlock = nullptr;
		if (it != renderingTSDF.end())
			renderBlock = it->second;
	}
	if (linearIdx >= 0 and linearIdx < 6)
	{
		ITMIndexDirectional index = ITMIndexDirectional(blockPos, TSDFDirection(linearIdx));

		auto it = tsdf.find(index);
		if (it == tsdf.end())
		{
			blocks[linearIdx] = nullptr;
		} else
		{
			blocks[linearIdx] = it->second;
		}
	}
	if (linearIdx >= 7 and linearIdx < 7 + 6 * N_DIRECTIONS)
	{
		int offset = linearIdx - 7;
		int directionIdx = offset / N_DIRECTIONS;
		int neighborIdx = offset % 6;

		Vector3s shift[6] =
			{
				Vector3s(1, 0, 0),
				Vector3s(-1, 0, 0),
				Vector3s(0, 1, 0),
				Vector3s(0, -1, 0),
				Vector3s(0, 0, 1),
				Vector3s(0, 0, -1)
			};
		ITMIndexDirectional index = ITMIndexDirectional(blockPos + shift[neighborIdx], TSDFDirection(directionIdx));
		auto it = tsdf.find(index);
		if (it == tsdf.end())
			neighborBlocks[directionIdx * 6 + neighborIdx] = nullptr;
		else
			neighborBlocks[directionIdx * 6 + neighborIdx] = it->second;
	}

	__syncthreads();

	if (not renderBlock)
		return;

	ITMVoxel& voxel = renderBlock[linearIdx];
	Vector3i voxelPosIdx = blockToVoxelPos(Vector3i(blockPos.x, blockPos.y, blockPos.z))
	                       + voxelOffsetToCoordinate(linearIdx);
	Vector3f voxelPos = voxelPosIdx.toFloat();
	Vector3f rayDirection = (voxelPos * voxelSize - (invM * Vector4f(0, 0, 0, 1)).toVector3()).normalised();

	Vector3f colorCombined(0, 0, 0);
	float sdfCombined = 0;
	float weightCombined = 0;
	float colorWeightCombined = 0;

	Vector3f colorCombinedNoGradient(0, 0, 0);
	Vector3f gradientCombined(0, 0, 0);
	float sdfCombinedNoGradient = 0;
	float weightCombinedNoGradient = 0;
	float colorWeightCombinedNoGradient = 0;

	Vector3f colorFreeSpace(0, 0, 0);
	Vector3f gradientFreeSpace(0, 0, 0);
	float freeSpaceWeight = 0;
	float freeSpaceSDF = 0;


	for (TSDFDirection_type directionIdx = 0; directionIdx < N_DIRECTIONS; directionIdx++)
	{
		ITMVoxel* block = blocks[directionIdx];
		if (not block)
			continue;

		float sdf = block[linearIdx].sdf;
		float w_depth = block[linearIdx].w_depth;

		Vector3f color;
		float color_w = 1;
		if (RENDER_DIRECTION_COLORS == 1)
			color = TSDFDirectionColor[directionIdx];
		else
			color = readFromSDF_color4u_uninterpolated(color_w, tsdf, voxelPos, maxW,
			                                           TSDFDirection(directionIdx)).toVector3();

		Vector3f gradient = computeGradient(block, neighborBlocks + directionIdx * 6, voxelOffsetToCoordinate(linearIdx),
		                                    voxelSize / mu);

		float weight =
			DirectionWeight(DirectionAngle(gradient, TSDFDirection(directionIdx)))
			* ORUtils::dot(gradient, -rayDirection)
			* w_depth;
		weight = MAX(weight, 0);

		float weightNoGradient = MAX(0,
		                             w_depth
		                             * ORUtils::dot(-rayDirection, TSDFDirectionVector[directionIdx])
//		                             * DirectionWeight(DirectionAngle(-rayDirection, TSDFDirection(directionIdx)))
		);
		weightCombinedNoGradient += weightNoGradient;
		sdfCombinedNoGradient += weightNoGradient * sdf;
		colorWeightCombinedNoGradient += weightNoGradient * color_w;
		colorCombinedNoGradient += weightNoGradient * color_w * color;

		sdfCombined += weight * sdf;
		weightCombined += weight;
		colorWeightCombined += weight * color_w;
		colorCombined += weight * color_w * color;
		gradientCombined += weight * gradient;
	}

	bool hasGradient = gradientCombined.x != 0 or gradientCombined.y != 0 or gradientCombined.z != 0;
	bool hasFreeSpaceGradient = gradientFreeSpace.x != 0 or gradientFreeSpace.y != 0 or gradientFreeSpace.z != 0;

	if (weightCombined > 0)
	{
		sdfCombined /= weightCombined;
		if (hasGradient) gradientCombined = gradientCombined.normalised();
	} else if (weightCombinedNoGradient > 0)
	{
		sdfCombined = sdfCombinedNoGradient / weightCombinedNoGradient;
		weightCombined = weightCombinedNoGradient;
		gradientCombined = Vector3f(0, 0, 0);
	} else
	{
		sdfCombined = 1;
		weightCombined = 0;
		gradientCombined = Vector3f(0, 0, 0);
	}

	if (colorWeightCombined > 0)
	{
		colorCombined /= colorWeightCombined;
	} else if (colorWeightCombinedNoGradient > 0)
	{
		colorCombined = colorCombinedNoGradient / colorWeightCombinedNoGradient;
		colorWeightCombined = colorWeightCombinedNoGradient;
	} else
	{
		colorCombined = Vector3f(0, 0, 0);
		colorWeightCombined = 0;
	}

	if (freeSpaceWeight > 0)
	{
		freeSpaceSDF /= freeSpaceWeight;
		colorFreeSpace /= freeSpaceWeight;
		if (hasFreeSpaceGradient)
			gradientFreeSpace = gradientFreeSpace.normalised();
	}

	if (freeSpaceWeight > 0
	    and ((hasGradient and hasFreeSpaceGradient
	          and dot(gradientFreeSpace, gradientCombined) <
	              0.707 // if same surface, use normal combination instead of free space (prevent dents in surface)
	          and dot(gradientFreeSpace, gradientCombined) > -0.707 // if opposite surface, don't carve
	         ) or (not hasGradient and not hasFreeSpaceGradient)))
	{
		voxel.sdf = ITMVoxel::floatToValue(freeSpaceSDF);
		voxel.w_depth = ITMVoxel::floatToWeight(freeSpaceWeight, maxW);
		voxel.clr = TO_UCHAR3(colorFreeSpace * 255.0f);
		voxel.w_color = ITMVoxel::floatToWeight(freeSpaceWeight, maxW);
	} else
	{
		voxel.sdf = ITMVoxel::floatToValue(sdfCombined);
		voxel.w_depth = ITMVoxel::floatToWeight(weightCombined, maxW);
		voxel.clr = TO_UCHAR3(colorCombined * 255.0f);
		voxel.w_color = ITMVoxel::floatToWeight(colorWeightCombined, maxW);
	}
}

__global__ void
combineDirectionalTSDFViewPoint_device(stdgpu::unordered_map<ITMIndex, ITMVoxel*> renderingTSDF,
                                       const stdgpu::unordered_map<ITMIndexDirectional, ITMVoxel*> tsdf,
                                       const ITMIndex* visibleBlocks, const stdgpu::index_t numVisibleBlocks,
                                       const Matrix4f invM, const float voxelSize, const float mu, const int maxW)
{
	__shared__ ITMVoxel* block;
	__shared__ ITMIndex blockPos;

	// cache tsdf lookup (once per block)
	int linearIdx = threadIdx.x;
	if (linearIdx == 0)
	{
		auto i = static_cast<stdgpu::index_t>(blockIdx.x);
		if (i >= numVisibleBlocks) return;
		blockPos = visibleBlocks[i];

		auto it = renderingTSDF.find(blockPos);
		if (it == renderingTSDF.end())
			return;

		block = it->second;
	}
	__syncthreads();

	ITMVoxel& voxel = block[linearIdx];
	Vector3i voxelPosIdx = blockToVoxelPos(Vector3i(blockPos.x, blockPos.y, blockPos.z))
	                       + voxelOffsetToCoordinate(linearIdx);
	Vector3f voxelPos = voxelPosIdx.toFloat();

	voxel = combineDirectionalTSDFViewPoint(voxelPos, tsdf, invM, voxelSize, mu, maxW);
}

} // namespace ITMLib
