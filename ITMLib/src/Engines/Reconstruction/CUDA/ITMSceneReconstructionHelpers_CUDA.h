//
// Created by Malte Splietker on 25.10.21.
//

#pragma once

#include <stdgpu/unordered_map.cuh>
#include <stdgpu/unordered_set.cuh>
#include "../Shared/ITMSceneReconstructionEngine_Shared.h"

namespace ITMLib
{

template<typename TIndex>
__global__
void rayCastCombine_device(
	stdgpu::unordered_map<TIndex, ITMVoxel*> tsdf,
	const stdgpu::unordered_map<TIndex, SummingVoxel*> summingVoxels,
	const TIndex* blocksList,
	const size_t noBlocks,
	const ITMSceneParams sceneParams
)
{
	size_t blockNo = blockIdx.x * gridDim.y + blockIdx.y;
	if (blockNo > noBlocks)
		return;
	TIndex idx = blocksList[blockNo];

	auto it_summing = summingVoxels.find(idx);
	auto it_tsdf = tsdf.find(idx);
	if (it_summing == summingVoxels.end() or it_tsdf == tsdf.end())
		return;

	const SummingVoxel* summingBlock = it_summing->second;
	ITMVoxel* tsdfBlock = it_tsdf->second;

	int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
	int locId = VoxelIndicesToOffset(x, y, z);

	rayCastCombine(tsdfBlock[locId], summingBlock[locId], sceneParams);
}

template<typename TIndex>
__global__
void rayCastUpdate_device(
	Vector2i imgSize_d, Vector2i imgSize_rgb, float* depth, Vector4f* depthNormals, Vector4u* rgb,
	const Matrix4f invM_d, const Matrix4f invM_rgb,
	const Vector4f invProjParams_d, const Vector4f projParams_rgb,
	const ITMFusionParams fusionParams,
	const ITMSceneParams sceneParams,
	stdgpu::unordered_map<TIndex, SummingVoxel*> summingVoxelMap
)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= imgSize_d.x or y >= imgSize_d.y)
		return;

	int idx = y * imgSize_d.x + x;

	Vector4f pt_sensor = Vector4f(reprojectImagePoint(x, y, depth[idx], invProjParams_d), 1);
	Vector4f normal_sensor = depthNormals[idx];

	Vector4f pt_world = invM_d * pt_sensor;
	Vector4f pt_rgb_camera = invM_rgb * pt_world;
	Vector2f cameraCoordsRGB = project(pt_rgb_camera.toVector3(), projParams_rgb);
	Vector4f color(0, 0, 0, 0);
	if ((cameraCoordsRGB.x >= 1) and (cameraCoordsRGB.x <= imgSize_rgb.x - 2)
	    and (cameraCoordsRGB.y >= 1) and (cameraCoordsRGB.y <= imgSize_rgb.y - 2))
		color = interpolateBilinear(rgb, cameraCoordsRGB, imgSize_rgb);

	rayCastUpdate(x, y, imgSize_d, imgSize_rgb, pt_sensor, normal_sensor, color, invM_d,
	              invProjParams_d, fusionParams, sceneParams, summingVoxelMap);
}

__global__
void rayCastUpdateFromCloud_device(
	Vector2i imgSize_d, Vector2i imgSize_rgb, Vector4f* points, Vector4f* normals, Vector4f* colors,
	const Matrix4f invM_d, const Matrix4f invM_rgb,
	const Vector4f invProjParams_d, const Vector4f projParams_rgb,
	const ITMFusionParams fusionParams,
	const ITMSceneParams sceneParams,
	stdgpu::unordered_map<ITMIndexDirectional, SummingVoxel*> summingVoxelMap
)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= imgSize_d.x or y >= imgSize_d.y)
		return;

	int idx = y * imgSize_d.x + x;

	rayCastUpdate(x, y, imgSize_d, imgSize_rgb, points[idx], normals[idx], colors[idx], invM_d,
	              invProjParams_d, fusionParams, sceneParams, summingVoxelMap);
}

template<typename TIndex>
__global__
void rayCastCarveSpace_device(
	Vector2i imgSize, float* depth, Vector4f* depthNormals,
	const Matrix4f invM_d,
	const Vector4f invProjParams_d, const Vector4f projParams_rgb,
	const ITMFusionParams fusionParams,
	const ITMSceneParams sceneParams,
	const stdgpu::unordered_map<TIndex, ITMVoxel*> tsdf,
	stdgpu::unordered_map<TIndex, SummingVoxel*> summingVoxelMap)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= imgSize.x or y >= imgSize.y)
		return;

	rayCastCarveSpace(x, y, imgSize, depth, depthNormals, invM_d,
	                  invProjParams_d, projParams_rgb, fusionParams, sceneParams,
	                  tsdf, summingVoxelMap);
}

template<bool stopMaxW, typename TIndex>
__global__ void voxelProjectionCarveSpace_device(
	const stdgpu::unordered_map<TIndex, ITMVoxel*> tsdf,
	stdgpu::unordered_map<TIndex, SummingVoxel*> summingVoxels,
	const TIndex* visibleBlocksList, const size_t noVisibleBlocks,
	const Vector4u* rgb, Vector2i rgbImgSize, const float* depth, const Vector4f* depthNormals,
	Vector2i depthImgSize, Matrix4f M_d, Matrix4f M_rgb, Vector4f projParams_d,
	Vector4f projParams_rgb, const ITMFusionParams fusionParams,
	const ITMSceneParams sceneParams)
{
	size_t blockNo = blockIdx.x * gridDim.y + blockIdx.y;
	int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;
	int locId = VoxelIndicesToOffset(x, y, z);

	if (blockNo > noVisibleBlocks)
		return;
	const TIndex& blockIdx = visibleBlocksList[blockNo];

	auto it = summingVoxels.find(blockIdx);
	if (it == summingVoxels.end())
		return;
	SummingVoxel& summingVoxel = it->second[locId];

	auto it_tsdf = tsdf.find(blockIdx);
	if (it_tsdf == tsdf.end())
		return;
	const ITMVoxel& voxel = it_tsdf->second[locId];

	if (stopMaxW) if (voxel.w_depth == sceneParams.maxW) return;

	float voxelSize = sceneParams.voxelSize;
	Vector4f pt_model;
	pt_model.x = (float) (blockIdx.getPosition().x + x) * voxelSize;
	pt_model.y = (float) (blockIdx.getPosition().y + y) * voxelSize;
	pt_model.z = (float) (blockIdx.getPosition().z + z) * voxelSize;
	pt_model.w = 1.0f;

	voxelProjectionCarveSpace(
		voxel, summingVoxel,
		blockIdx.getDirection(),
		pt_model, M_d, projParams_d, M_rgb, projParams_rgb,
		fusionParams, sceneParams, depth, depthNormals,
		depthImgSize, rgb, rgbImgSize);
}

template<bool stopMaxW, typename TIndex>
__global__ void integrateIntoScene_device(stdgpu::unordered_map<TIndex, ITMVoxel*> tsdf, TIndex* visibleEntries,
                                          const Vector4u* rgb, Vector2i rgbImgSize, const float* depth,
                                          const Vector4f* depthNormals, Vector2i depthImgSize,
                                          Matrix4f M_d, Matrix4f M_rgb, Vector4f projParams_d, Vector4f projParams_rgb,
                                          float _voxelSize, const ITMFusionParams fusionParams,
                                          const ITMSceneParams sceneParams)
{
	const TIndex& index = visibleEntries[blockIdx.x];
	auto it = tsdf.find(index);
	if (it == tsdf.end())
		return;
	ITMVoxel* localVoxelBlock = it->second;
	int x = threadIdx.x, y = threadIdx.y, z = threadIdx.z;

	int locId = VoxelIndicesToOffset(x, y, z);

	if (stopMaxW) if (localVoxelBlock[locId].w_depth == sceneParams.maxW) return;

	Vector4f pt_model = Vector4f(voxelIdxToWorldPos(index.getPosition() * SDF_BLOCK_SIZE + Vector3s(x, y, z), _voxelSize),
	                             1.0f);

	std::conditional<ITMVoxel::hasColorInformation, ComputeUpdatedVoxelInfo<true, ITMVoxel>, ComputeUpdatedVoxelInfo<false, ITMVoxel>>::type::compute(
		localVoxelBlock[locId], index.getDirection(),
		pt_model, M_d, projParams_d, M_rgb, projParams_rgb, fusionParams, sceneParams, depth, depthNormals,
		depthImgSize, rgb, rgbImgSize);
}

template<typename TIndex>
__global__ void findAllocationBlocks_device(stdgpu::unordered_set<TIndex> allocationBlocks,
                                            const float* depth, const Vector4f* depthNormal,
                                            Matrix4f invM_d, Vector4f invProjParams_d, float mu, Vector2i imgSize,
                                            float voxelSize,
                                            float viewFrustum_min, float viewFrustum_max,
                                            const ITMFusionParams fusionParams)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > imgSize.x - 1 || y > imgSize.y - 1) return;

	findAllocationBlocks(allocationBlocks, x, y, depth, depthNormal, invM_d, invProjParams_d, mu, imgSize, voxelSize,
	                     viewFrustum_min, viewFrustum_max, fusionParams);
}

}
