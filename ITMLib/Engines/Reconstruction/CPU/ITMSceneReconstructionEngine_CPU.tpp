// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include <unordered_map>
#include <unordered_set>
#include "ITMSceneReconstructionEngine_CPU.h"

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <ITMLib/Engines/Reconstruction/Shared/ITMFusionWeight.hpp>
#include <ITMLib/Engines/Reconstruction/Shared/ITMSceneReconstructionEngine_Shared.h>
#include <ITMLib/Objects/RenderStates/ITMRenderState.h>
#include <ITMLib/Objects/Scene/TSDF_CPU.h>

using namespace ITMLib;

template<typename TIndex>
ITMSceneReconstructionEngine_CPU<TIndex>::ITMSceneReconstructionEngine_CPU(
	const std::shared_ptr<const ITMLibSettings>& settings)
	: ITMSceneReconstructionEngine<TIndex>(settings, MEMORYDEVICE_CPU)
{
	summingVoxelMap = new TSDF_CPU<TIndex, SummingVoxel>();
}

template<typename TIndex>
ITMSceneReconstructionEngine_CPU<TIndex>::~ITMSceneReconstructionEngine_CPU()
{
	delete summingVoxelMap;
}

template<typename TIndex>
void ITMSceneReconstructionEngine_CPU<TIndex>::ResetScene(Scene* scene)
{
	scene->Clear();
}

template<typename TIndex>
void ITMSceneReconstructionEngine_CPU<TIndex>::IntegrateIntoSceneRayCasting(
	Scene* scene, const ITMView* view, const ITMTrackingState* trackingState)
{
	ITMTimer timer;
	timer.Tick();

	const Matrix4f invM_d = trackingState->pose_d->GetInvM();
	const Matrix4f M_rgb = trackingState->pose_d->GetM();

	const Vector4f& projParams_d = view->calib.intrinsics_d.projectionParamsSimple.all;
	const Vector4f& projParams_rgb = view->calib.intrinsics_rgb.projectionParamsSimple.all;
	const Vector2i& imgSize_rgb = view->calib.intrinsics_rgb.imgSize;

	const float* depth = view->depth->GetData(MEMORYDEVICE_CPU);
	const Vector4f* depthNormals = view->depthNormal->GetData(MEMORYDEVICE_CPU);
	const Vector4u* rgb = view->rgb->GetData(MEMORYDEVICE_CPU);

	const Vector4f invProjParams_d = invertProjectionParams(projParams_d);

	auto& tsdf = GetTSDF(scene)->getMap();
	auto& summingVoxels = this->summingVoxelMap->getMap();
	TIndex* visibleEntries = this->allocationFusionBlocksList->GetData(MEMORYDEVICE_CPU);

	/// 1. Initialize summing voxel map (allocate and reset)
	summingVoxelMap->resize(this->noAllocationBlocks);
	summingVoxelMap->allocate(this->allocationFusionBlocksList->GetData(MEMORYDEVICE_CPU), this->noAllocationBlocks);

	/// 2. Ray trace every pixel, sum up results
	for (int y = 0; y < view->depth->noDims.y; y++)
		for (int x = 0; x < view->depth->noDims.x; x++)
		{
			int idx = y * view->depth->noDims.x + x;
			Vector4f pt_sensor = Vector4f(reprojectImagePoint(x, y, depth[idx], invProjParams_d), 1);
			Vector4f normal_sensor = depthNormals[idx];

			Vector4f pt_world = invM_d * pt_sensor;
			Vector4f pt_rgb_camera = M_rgb * pt_world;
			Vector2f cameraCoordsRGB = project(pt_rgb_camera.toVector3(), projParams_rgb);
			Vector4f color(0, 0, 0, 0);
			if ((cameraCoordsRGB.x >= 1) and (cameraCoordsRGB.x <= imgSize_rgb.x - 2)
			    and (cameraCoordsRGB.y >= 1) and (cameraCoordsRGB.y <= imgSize_rgb.y - 2))
				color = interpolateBilinear(rgb, cameraCoordsRGB, imgSize_rgb);
			rayCastUpdate(x, y, view->depth->noDims, view->rgb->noDims, pt_sensor, normal_sensor, color, invM_d,
			              invProjParams_d, this->settings->fusionParams, this->settings->sceneParams, summingVoxels);
		}
	this->timeStats.fusion += timer.Tock();

	/// 3. Ray trace space carve every pixel, sum up results
	timer.Tick();
	if (this->settings->fusionParams.useSpaceCarving)
	{
		if (this->settings->fusionParams.carvingMode == CarvingMode::CARVINGMODE_RAY_CASTING)
		{
			for (int y = 0; y < view->depth->noDims.y; y++)
				for (int x = 0; x < view->depth->noDims.x; x++)
				{
					rayCastCarveSpace(x, y, view->depth->noDims, depth, depthNormals, invM_d,
					                  invProjParams_d, projParams_rgb, this->settings->fusionParams,
					                  this->settings->sceneParams, tsdf, summingVoxels);
				}
		} else
		{
			bool stopIntegratingAtMaxW = scene->sceneParams->stopIntegratingAtMaxW;
			float voxelSize = scene->sceneParams->voxelSize;
			Matrix4f M_d = trackingState->pose_d->GetM();
			Matrix4f M_rgb = view->calib.trafo_rgb_to_depth.calib_inv * M_d;
			Vector2i depthImgSize = view->depth->noDims;
			Vector2i rgbImgSize = view->rgb->noDims;
			for (size_t entryId = 0; entryId < this->noFusionBlocks; entryId++)
			{
				TIndex block = visibleEntries[entryId];

				auto it = summingVoxels.find(block);
				if (it == summingVoxels.end())
					continue;

				auto it_tsdf = tsdf.find(block);
				if (it_tsdf == tsdf.end())
					continue;

				for (int z = 0; z < SDF_BLOCK_SIZE; z++)
					for (int y = 0; y < SDF_BLOCK_SIZE; y++)
						for (int x = 0; x < SDF_BLOCK_SIZE; x++)
						{
							int locId = VoxelIndicesToOffset(x, y, z);

							SummingVoxel& summingVoxel = it->second[locId];
							const ITMVoxel& voxel = it_tsdf->second[locId];

							if (stopIntegratingAtMaxW) if (voxel.w_depth == this->settings->sceneParams.maxW) continue;

							Vector4f pt_model;
							pt_model.x = (float) (block.getPosition().x + x) * voxelSize;
							pt_model.y = (float) (block.getPosition().y + y) * voxelSize;
							pt_model.z = (float) (block.getPosition().z + z) * voxelSize;
							pt_model.w = 1.0f;

							voxelProjectionCarveSpace(
								voxel, summingVoxel,
								block.getDirection(), pt_model, M_d, projParams_d, M_rgb, projParams_rgb,
								this->settings->fusionParams, this->settings->sceneParams, depth, depthNormals,
								depthImgSize, rgb, rgbImgSize);
						}
			}
		}
	}
	this->timeStats.carving += timer.Tock();

	/// 4. Collect per-voxel summation values, fuse into voxels
	timer.Tick();
	for (size_t entryId = 0; entryId < this->noAllocationBlocks; entryId++)
	{
		TIndex block = visibleEntries[entryId];

		auto it = summingVoxels.find(block);
		if (it == summingVoxels.end())
			continue;

		auto it_tsdf = tsdf.find(block);
		if (it_tsdf == tsdf.end())
			continue;

		const SummingVoxel* summingBlock = it->second;
		ITMVoxel* tsdfBlock = it_tsdf->second;

		for (int locId = 0; locId < SDF_BLOCK_SIZE3; locId++)
		{
			rayCastCombine(tsdfBlock[locId], summingBlock[locId], this->settings->sceneParams);
		}
	}
	this->timeStats.fusion += timer.Tock();
}

template<typename TIndex>
void ITMSceneReconstructionEngine_CPU<TIndex>::IntegrateIntoSceneVoxelProjection(
	Scene* scene, const ITMView* view,
	const ITMTrackingState* trackingState)
{
	ITMTimer timer;
	timer.Tick();

	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;

	Matrix4f M_d, M_rgb;
	Vector4f projParams_d, projParams_rgb;

	M_d = trackingState->pose_d->GetM();
	if (ITMVoxel::hasColorInformation) M_rgb = view->calib.trafo_rgb_to_depth.calib_inv * M_d;

	projParams_d = view->calib.intrinsics_d.projectionParamsSimple.all;
	projParams_rgb = view->calib.intrinsics_rgb.projectionParamsSimple.all;

	float* depth = view->depth->GetData(MEMORYDEVICE_CPU);
	Vector4f* depthNormals = nullptr;
	if (this->settings->fusionParams.useWeighting or
	    this->settings->fusionParams.fusionMetric == FUSIONMETRIC_POINT_TO_PLANE)
		depthNormals = view->depthNormal->GetData(MEMORYDEVICE_CPU);
	Vector4u* rgb = view->rgb->GetData(MEMORYDEVICE_CPU);

	bool stopIntegratingAtMaxW = scene->sceneParams->stopIntegratingAtMaxW;
	//bool approximateIntegration = !trackingState->requiresFullRendering;

	std::unordered_map<TIndex, ITMVoxel*> tsdf = GetTSDF(scene)->getMap();

	TIndex* visibleEntries = this->allocationFusionBlocksList->GetData(MEMORYDEVICE_CPU);

#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
	for (size_t entryId = 0; entryId < this->noFusionBlocks; entryId++)
	{
		const TIndex& index = visibleEntries[entryId];
		auto it = tsdf.find(index);
		if (it == tsdf.end())
			continue;
		ITMVoxel* localVoxelBlock = it->second;

		for (int z = 0; z < SDF_BLOCK_SIZE; z++)
			for (int y = 0; y < SDF_BLOCK_SIZE; y++)
				for (int x = 0; x < SDF_BLOCK_SIZE; x++)
				{
					int locId = VoxelIndicesToOffset(x, y, z);

					if (stopIntegratingAtMaxW) if (localVoxelBlock[locId].w_depth == scene->sceneParams->maxW) continue;

					Vector4f pt_model = Vector4f(voxelIdxToWorldPos(index.getPosition() * SDF_BLOCK_SIZE + Vector3s(x, y, z),
					                                                scene->sceneParams->voxelSize), 1.0f);

					std::conditional<ITMVoxel::hasColorInformation, ComputeUpdatedVoxelInfo<true, ITMVoxel>, ComputeUpdatedVoxelInfo<false, ITMVoxel>>::type::compute(
						localVoxelBlock[locId], index.getDirection(),
						pt_model, M_d, projParams_d, M_rgb, projParams_rgb, this->settings->fusionParams,
						this->settings->sceneParams,
						depth, depthNormals, depthImgSize, rgb, rgbImgSize);
				}
	}
	this->timeStats.fusion += timer.Tock();
}

template<typename TIndex>
void ITMSceneReconstructionEngine_CPU<TIndex>::AllocateSceneFromDepth(Scene* scene, const ITMView* view,
                                                                      const ITMTrackingState* trackingState)
{
	ITMTimer timer;
	timer.Tick();
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f invM_d = trackingState->pose_d->GetInvM();
	Vector4f invProjParams_d = invertProjectionParams(view->calib.intrinsics_d.projectionParamsSimple.all);

	float mu = scene->sceneParams->mu;

	float* depth = view->depth->GetData(MEMORYDEVICE_CPU);
	Vector4f* depthNormal = view->depthNormal->GetData(MEMORYDEVICE_CPU);

	auto tsdf = GetTSDF(scene);
	if (tsdf->size() >= tsdf->allocatedBlocksMax)
	{
		printf("No more free blocks. Allocation stopped.\n");
	}

	std::unordered_set<TIndex> allocationBlocks;
	allocationBlocks.reserve(1e6);
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
	for (int locId = 0; locId < depthImgSize.x * depthImgSize.y; locId++)
	{
		int y = locId / depthImgSize.x;
		int x = locId - y * depthImgSize.x;

		findAllocationBlocks(allocationBlocks, x, y, depth, depthNormal, invM_d, invProjParams_d, mu, depthImgSize,
		                     voxelSize,
		                     scene->sceneParams->viewFrustum_min, scene->sceneParams->viewFrustum_max,
		                     this->settings->fusionParams);
	}

	size_t N = allocationBlocks.size();
	if (N > this->allocationFusionBlocksList->dataSize)
		this->allocationFusionBlocksList->Resize(N);
	this->noAllocationBlocks = N;
	thrust::copy(allocationBlocks.begin(), allocationBlocks.end(),
	             this->allocationFusionBlocksList->GetData(MEMORYDEVICE_CPU));
	tsdf->allocate(this->allocationFusionBlocksList->GetData(MEMORYDEVICE_CPU), N);

	this->timeStats.allocation += timer.Tock();
}

template<typename TIndex>
void ITMSceneReconstructionEngine_CPU<TIndex>::FindVisibleBlocks(const Scene* scene, const ORUtils::SE3Pose* pose,
                                                                 const ITMIntrinsics* intrinsics,
                                                                 ITMRenderState* renderState)
{
	ITMTimer timer;
	timer.Tick();

	auto& tsdf = GetTSDF(scene)->getMap();

	std::unordered_set<ITMIndex> visibleBlocks;
	visibleBlocks.reserve(MAX(renderState->AllocatedSize(), this->allocationFusionBlocksList->dataSize * 2));

	if (this->noFusionBlocks > 0.8 * this->allocationFusionBlocksList->dataSize)
	{
		this->allocationFusionBlocksList->Resize(2 * this->allocationFusionBlocksList->dataSize);
	}
	unsigned long long fusionBlocksCounter = 0;
	thrust::for_each(thrust::host, tsdf.begin(), tsdf.end(),
	                 findVisibleBlocksFunctor<TIndex, std::unordered_set>(
		                 visibleBlocks,
		                 this->allocationFusionBlocksList->GetData(MEMORYDEVICE_CPU),
		                 this->allocationFusionBlocksList->dataSize,
		                 &fusionBlocksCounter,
		                 pose->GetM(),
		                 intrinsics->projectionParamsSimple.all,
		                 renderState->renderingRangeImage->noDims,
		                 *(scene->sceneParams)));

	renderState->Resize(visibleBlocks.size());
	thrust::copy(visibleBlocks.begin(), visibleBlocks.end(), renderState->GetVisibleBlocks());
	renderState->noVisibleEntries = visibleBlocks.size();
	this->noFusionBlocks = MIN(fusionBlocksCounter, this->allocationFusionBlocksList->dataSize);

	this->timeStats.buildingVisibilityList += timer.Tock();
}

template<typename TIndex>
TSDF_CPU<TIndex, ITMVoxel>* ITMSceneReconstructionEngine_CPU<TIndex>::GetTSDF(const Scene* scene)
{
	if (this->directional)
		return dynamic_cast<TSDF_CPU<TIndex, ITMVoxel>*>(scene->tsdfDirectional);
	else
		return dynamic_cast<TSDF_CPU<TIndex, ITMVoxel>*>(scene->tsdf);
}
