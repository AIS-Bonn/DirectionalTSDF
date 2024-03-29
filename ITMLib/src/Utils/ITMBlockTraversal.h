//
// Created by Malte Splietker on 15.05.19.
//

#pragma once

#include "ITMLib/Utils/ITMGeometry.h"

namespace ITMLib
{

struct BlockTraversal
{
	/**
	 * @param origin Ray origin
	 * @param dir Ray direction
	 * @param truncation_distance Truncation distance
	 * @param block_size Size of the blocks to traverse
	 * @param round_to_nearest Whether to round the indices to the nearest responsible index
	 */
	_CPU_AND_GPU_CODE_
	BlockTraversal(const Vector3f& origin,
	               const Vector3f& dir,
	               const float truncation_distance,
	               const float block_size,
	               const bool round_to_nearest = true
	)
		: origin(origin), direction(normalize(dir)), truncation_distance(truncation_distance),
		  block_size(block_size),
		  round_to_nearest(round_to_nearest),
		  step_size(Vector3i(dir.x > 0 ? 1 : -1,
		                     dir.y > 0 ? 1 : -1,
		                     dir.z > 0 ? 1 : -1)),
		  tDelta(direction.x == 0 ? MAXFLOAT : std::abs(block_size / direction.x),
		         direction.y == 0 ? MAXFLOAT : std::abs(block_size / direction.y),
		         direction.z == 0 ? MAXFLOAT : std::abs(block_size / direction.z))
	{
		if (ORUtils::length(direction) == 0)
		{
			printf("ERROR: direction of block traversal must not be 0!\n");
			distance = truncation_distance;
			return;
		}

		Vector3f val = WorldToBlockf(origin);
		Vector3f inner_block_offset = Vector3f((val.x - floor(val.x)),
		                                       (val.y - floor(val.y)),
		                                       (val.z - floor(val.z)))
		                              * block_size;

		// Initialize with distance along ray to first x/y/z block borders
		tMax = Vector3f(
			std::abs(
				direction.x > 0 ?
				(block_size - inner_block_offset.x) / direction.x :
				direction.x == 0 ? MAXFLOAT : inner_block_offset.x / direction.x),
			std::abs(
				direction.y > 0 ?
				(block_size - inner_block_offset.y) / direction.y :
				direction.y == 0 ? MAXFLOAT : inner_block_offset.y / direction.y),
			std::abs(
				direction.z > 0 ?
				(block_size - inner_block_offset.z) / direction.z :
				direction.z == 0 ? MAXFLOAT : inner_block_offset.z / direction.z)
		);
		next_block = WorldToBlocki(origin);
		distance = 0;
	}

	_CPU_AND_GPU_CODE_
	inline Vector3f WorldToBlockf(const Vector3f world_pos)
	{
		return world_pos / block_size;
	}

	_CPU_AND_GPU_CODE_
	inline Vector3i WorldToBlocki(const Vector3f world_pos)
	{
		const Vector3f p = WorldToBlockf(world_pos);
		if (round_to_nearest)
		{
			Vector3f sign(p.x > 0 ? 1 : -1, p.y > 0 ? 1 : -1, p.z > 0 ? 1 : -1);
			return (p + sign * 0.5f).toInt();
		}
		Vector3i idx = p.toInt();
		if (p.x < 0) idx.x -= 1;
		if (p.y < 0) idx.y -= 1;
		if (p.z < 0) idx.z -= 1;
		return idx;
	}

	_CPU_AND_GPU_CODE_
	inline Vector3f BlockToWorld(const Vector3i& blockPos)
	{
		return blockPos.toFloat() * block_size;
	}

	_CPU_AND_GPU_CODE_
	bool HasNextBlock()
	{
		return distance < truncation_distance;
	}

	_CPU_AND_GPU_CODE_
	void Step()
	{
		// Distance along the ray to next block
		distance = fminf(fminf(tMax.x, tMax.y), tMax.z);

//		if (tMax.x < tMax.y)
//		{
//			if (tMax.x < tMax.z)
//			{
//				next_block.x += step_size.x;
//				tMax.x += tDelta.x;
//			} else
//			{
//				next_block.z += step_size.z;
//				tMax.z += tDelta.z;
//			}
//		} else
//		{
//			if (tMax.y < tMax.z)
//			{
//				next_block.y += step_size.y;
//				tMax.y += tDelta.y;
//			} else
//			{
//				next_block.z += step_size.z;
//				tMax.z += tDelta.z;
//			}
//		}

		// CUDA optimized version of the above (no branching)
		bool c_xy = tMax.x < tMax.y;
		bool c_xz = tMax.x < tMax.z;
		bool c_yz = tMax.y < tMax.z;
		next_block.x += int(c_xy && c_xz) * step_size.x;
		tMax.x += float(c_xy && c_xz) * tDelta.x;

		next_block.y += int(!c_xy && c_yz) * step_size.y;
		tMax.y += float(!c_xy && c_yz) * tDelta.y;

		next_block.z += int((c_xy && !c_xz) || (!c_xy && !c_yz)) * step_size.z;
		tMax.z += float((c_xy && !c_xz) || (!c_xy && !c_yz)) * tDelta.z;
	}

	_CPU_AND_GPU_CODE_
	Vector3i GetNextBlock()
	{
		Vector3i current_block = next_block;

		Step();
		if (round_to_nearest)
		{
			while (current_block == WorldToBlocki(origin + distance * direction) and distance < truncation_distance)
			{ // Advance to prevent doubles
				Step();
			}
			next_block = WorldToBlocki(origin + distance * direction);
		}

		return current_block;
	}

	const Vector3f origin;
	const Vector3f direction;
	const float truncation_distance;
	const float block_size;
	const bool round_to_nearest;

	const Vector3i step_size;

	/** Distance along the ray to cover one block size in x/y/z direction, respectively */
	const Vector3f tDelta;

	/** Distance along the ray of the next boundary crossing in x/y/z direction, respectively */
	Vector3f tMax;

	/** Traversed distance along the ray */
	float distance;

	/** Next block (integer coordinates) */
	Vector3i next_block;
};

} // namespace ITMLib
