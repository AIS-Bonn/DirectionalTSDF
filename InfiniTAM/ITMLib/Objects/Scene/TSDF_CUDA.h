//
// Created by Malte Splietker on 08.04.21.
//

#pragma once

#include <ORUtils/CUDADefines.h>
#include <ITMLib/Core/ITMConstants.h>
#include <ITMLib/Objects/Scene/TSDF.h>
#include <ITMLib/Utils/ITMCUDAUtils.h>
#include <ITMLib/Utils/ITMGeometry.h>
#include <stdgpu/unordered_map.cuh>

namespace ITMLib
{

template<typename TVoxel>
__global__
void clearVoxels_device(TVoxel* voxels)
{
	int locId = threadIdx.x;
	int entryId = blockIdx.x;

	voxels[entryId * SDF_BLOCK_SIZE3 + locId] = TVoxel();
}


template <typename TVoxel, typename TIndex>
class TSDF_CUDA : public TSDF<TVoxel, TIndex, stdgpu::unordered_map>
{
public:
	~TSDF_CUDA()
	{
//		ORcudaSafeCall(cudaFree(this->voxels));
//		if (this->map.bucket_count() > 0)
//			stdgpu::unordered_map<TIndex, TVoxel*>::destroyDeviceObject(this->map);
	}

	void Resize(size_t newSize) override
	{
		if (newSize <= 0)
			return;
		this->allocationSize = newSize;

		if (this->voxels)
			destroyDeviceArray(this->voxels);
		if (this->map.bucket_count() > 0)
			stdgpu::unordered_map<TIndex, TVoxel*>::destroyDeviceObject(this->map);

		this->voxels = createDeviceArray(newSize * SDF_BLOCK_SIZE3, TVoxel());
		this->map = stdgpu::unordered_map<TIndex, TVoxel*>::createDeviceObject(
			newSize * 3.0); // factor 3 to prevent hash bucket overflow

		clearVoxels_device << < this->allocationSize, SDF_BLOCK_SIZE3 >> > (this->voxels);
		ORcudaKernelCheck;
	}

	explicit TSDF_CUDA(size_t size)
	{
		Resize(size);
	}
};

template<typename TVoxel, typename TIndex>
_CPU_AND_GPU_CODE_
inline TVoxel readVoxel(bool& found, const stdgpu::unordered_map<TIndex, TVoxel*>& tsdf,
                        const Vector3i& voxelIdx, const TSDFDirection direction = TSDFDirection::NONE)
{
#ifndef __CUDA_ARCH__
	fprintf(stderr, "error: no CPU access on TSDF_CUDA\n");
		return TVoxel();
#else
	TIndex index;
	unsigned short linearIdx;
	voxelIdxToIndexAndOffset(index, linearIdx, voxelIdx, direction);

	auto it = tsdf.find(index);
	if (it == tsdf.end())
	{
		found = false;
		return TVoxel();
	}

	found = true;
	return it->second[linearIdx];
#endif
}

} // namespace ITMlib