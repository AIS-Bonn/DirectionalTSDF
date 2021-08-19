//
// Created by Malte Splietker on 04.06.19.
//

#pragma once

#include "ITMLib/Core/ITMConstants.h"

namespace ITMLib
{

_CPU_AND_GPU_CODE_
inline Vector3i voxelToBlockPos(Vector3i voxelPos)
{
	voxelPos.x -= (voxelPos.x < 0) * (SDF_BLOCK_SIZE - 1);
	voxelPos.y -= (voxelPos.y < 0) * (SDF_BLOCK_SIZE - 1);
	voxelPos.z -= (voxelPos.z < 0) * (SDF_BLOCK_SIZE - 1);
	return voxelPos / SDF_BLOCK_SIZE;
}

_CPU_AND_GPU_CODE_
inline Vector3i blockToVoxelPos(const Vector3i& blockPos)
{
	return blockPos * SDF_BLOCK_SIZE;
}

/**
 * Convertes the voxel indices with a block into a linear array offset for accessing.
 * @param x
 * @param y
 * @param z
 * @return
 */
_CPU_AND_GPU_CODE_
inline int VoxelIndicesToOffset(int x, int y, int z)
{
	return x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
}

/**
 * Converts voxel position to block position and array offset (within the block's voxel array)
 * @param voxelPos
 * @param blockPos
 * @param offset
 */
_CPU_AND_GPU_CODE_ inline void voxelToBlockPosAndOffset(
	const Vector3i& voxelPos, Vector3i& blockPos, unsigned short &offset)
{
	blockPos = voxelToBlockPos(voxelPos);

	offset = voxelPos.x
					 + (voxelPos.y - blockPos.x) * SDF_BLOCK_SIZE
					 + (voxelPos.z - blockPos.y) * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE
					 - blockPos.z * SDF_BLOCK_SIZE3;
}

_CPU_AND_GPU_CODE_ inline Vector3i voxelOffsetToCoordinate(int &offset)
{
	Vector3i coordinate;
	coordinate.z = offset / (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);
	coordinate.y = offset / SDF_BLOCK_SIZE - coordinate.z * SDF_BLOCK_SIZE;
	coordinate.x = offset - coordinate.y * SDF_BLOCK_SIZE - coordinate.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
	return coordinate;
}

_CPU_AND_GPU_CODE_ inline Vector3f voxelIdxToWorldPos(const Vector3i &voxelIdx, const float voxelSize)
{
	return voxelIdx.toFloat() * voxelSize;
}

_CPU_AND_GPU_CODE_ inline Vector3i worldPosToVoxelIdx(const Vector3f &worldPos, const float voxelSize)
{
	return ((worldPos + voxelSize * Vector3f(0.5, 0.5, 0.5)) / voxelSize).toInt();
}

} // namespace ITMLib
