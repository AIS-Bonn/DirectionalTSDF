// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include <unordered_map>
#include "ITMSceneReconstructionEngine_CPU.h"
#include "ITMSummingVoxelMap_CPU.h"

#include "ITMLib/Objects/RenderStates/ITMRenderState_VH.h"
#include "ITMLib/Engines/Reconstruction/Shared/ITMFusionWeight.hpp"
#include "ITMLib/Engines/Reconstruction/Shared/ITMSceneReconstructionEngine_Shared.h"
using namespace ITMLib;

ITMSceneReconstructionEngine_CPU::ITMSceneReconstructionEngine_CPU(std::shared_ptr<const ITMLibSettings> settings)
	: ITMSceneReconstructionEngine(settings)
	{
		int noTotalEntries = ITMVoxelBlockHash::noTotalEntries;
	entriesAllocType = new ORUtils::MemoryBlock<HashEntryAllocType>(noTotalEntries, MEMORYDEVICE_CPU);
	blockCoords = new ORUtils::MemoryBlock<Vector4s>(noTotalEntries, MEMORYDEVICE_CPU);
	blockDirections = new ORUtils::MemoryBlock<TSDFDirection>(noTotalEntries, MEMORYDEVICE_CPU);
	summingVoxelMap = new SummingVoxelMap_CPU;
}

ITMSceneReconstructionEngine_CPU::~ITMSceneReconstructionEngine_CPU(void)
{
	delete entriesAllocType;
	delete blockCoords;
	delete blockDirections;
	delete summingVoxelMap;
}

void ITMSceneReconstructionEngine_CPU::ResetScene(Scene *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

	ITMVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
	for (int i = 0; i < numBlocks * blockSize; ++i) voxelBlocks_ptr[i] = ITMVoxel();
	int *vbaAllocationList_ptr = scene->localVBA.GetAllocationList();
	for (int i = 0; i < numBlocks; ++i) vbaAllocationList_ptr[i] = i;
	scene->localVBA.lastFreeBlockId = numBlocks - 1;

	ITMHashEntry tmpEntry;
	memset(&tmpEntry, 0, sizeof(ITMHashEntry));
	tmpEntry.ptr = -2;
	ITMHashEntry *hashEntry_ptr = scene->index.GetEntries();
	for (int i = 0; i < scene->index.noTotalEntries; ++i) hashEntry_ptr[i] = tmpEntry;
	int *excessList_ptr = scene->index.GetExcessAllocationList();
	for (int i = 0; i < SDF_EXCESS_LIST_SIZE; ++i) excessList_ptr[i] = i;

	scene->index.SetLastFreeExcessListId(SDF_EXCESS_LIST_SIZE - 1);
}


void ITMSceneReconstructionEngine_CPU::IntegrateIntoSceneRayCasting(
	Scene* scene, const ITMView* view, const ITMTrackingState* trackingState,
	const ITMRenderState* renderState)
{
	ITMTimer timer;
	timer.Tick();

	Matrix4f invM_d = trackingState->pose_d->GetInvM();

	Vector4f projParams_d = view->calib.intrinsics_d.projectionParamsSimple.all;
	Vector4f projParams_rgb = view->calib.intrinsics_rgb.projectionParamsSimple.all;

	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
	Vector4f *depthNormals = view->depthNormal->GetData(MEMORYDEVICE_CPU);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CPU);
	ITMVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	ITMHashEntry *hashTable = scene->index.GetEntries();

	Vector4f invProjParams_d = invertProjectionParams(projParams_d);
	const ITMRenderState_VH *renderState_vh = dynamic_cast<const ITMRenderState_VH*>(renderState);

	/// 1. Initialize summing voxel map (allocate and reset)
	summingVoxelMap->Init(hashTable, renderState_vh->GetVisibleEntryIDs(), renderState_vh->noVisibleEntries);

	/// 2. Ray trace every pixel, sum up results
	for (int y = 0; y < view->depth->noDims.y; y++) for (int x = 0; x < view->depth->noDims.x; x++)
	{
//		rayCastUpdate(x, y, view->depth->noDims, view->rgb->noDims, depth, depthNormals, rgb, invM_d, trackingState->pose_d->GetM(),
//		              invProjParams_d, projParams_rgb,
//		              this->settings->fusionParams, this->settings->sceneParams,
//		              summingVoxelMap->getMap());
	}
	this->timeStats.fusion += timer.Tock();

	/// 3. Ray trace space carve every pixel, sum up results
	int noVisibleEntries = renderState_vh->noVisibleEntries;
	const int *visibleEntryIds = renderState_vh->GetVisibleEntryIDs();
	if (this->settings->fusionParams.useSpaceCarving)
	{
		timer.Tick();
		if (this->settings->fusionParams.carvingMode == CarvingMode::CARVINGMODE_RAY_CASTING)
		{
			for (int y = 0; y < view->depth->noDims.y; y++) for (int x = 0; x < view->depth->noDims.x; x++)
			{
//				rayCastCarveSpace(x, y, view->depth->noDims, depth, depthNormals, invM_d,
//				                  invProjParams_d, projParams_rgb,
//				                  this->settings->fusionParams, this->settings->sceneParams,
//				                  hashTable, summingVoxelMap->getMap(), localVBA);
			}
		} else
		{
			bool stopIntegratingAtMaxW = scene->sceneParams->stopIntegratingAtMaxW;
			int maxW = scene->sceneParams->maxW;
			float voxelSize = scene->sceneParams->voxelSize;
			Matrix4f M_d = trackingState->pose_d->GetM();
			Matrix4f M_rgb = view->calib.trafo_rgb_to_depth.calib_inv * M_d;
			Vector2i depthImgSize = view->depth->noDims;
			Vector2i rgbImgSize = view->rgb->noDims;
			for (int entryId = 0; entryId < noVisibleEntries; entryId++)
			{
				const ITMHashEntry &currentHashEntry = hashTable[visibleEntryIds[entryId]];

				if (currentHashEntry.ptr < 0) continue;

				Vector3i globalPos = blockToVoxelPos(Vector3i(currentHashEntry.pos.x, currentHashEntry.pos.y, currentHashEntry.pos.z));

				ITMVoxel *localVoxelBlock = &(localVBA[currentHashEntry.ptr * (SDF_BLOCK_SIZE3)]);
				SummingVoxel *localRayCastingSum = &(summingVoxelMap->getVoxels()[currentHashEntry.ptr * (SDF_BLOCK_SIZE3)]);

				for (int z = 0; z < SDF_BLOCK_SIZE; z++) for (int y = 0; y < SDF_BLOCK_SIZE; y++) for (int x = 0; x < SDF_BLOCK_SIZE; x++)
						{
							Vector4f pt_model; int locId;

							locId = VoxelIndicesToOffset(x, y, z);

							if (stopIntegratingAtMaxW) if (localVoxelBlock[locId].w_depth == maxW) continue;
							//if (approximateIntegration) if (localVoxelBlock[locId].w_depth != 0) continue;

							pt_model.x = (float)(globalPos.x + x) * voxelSize;
							pt_model.y = (float)(globalPos.y + y) * voxelSize;
							pt_model.z = (float)(globalPos.z + z) * voxelSize;
							pt_model.w = 1.0f;

//							voxelProjectionCarveSpace(
//								localVoxelBlock[locId], localRayCastingSum[locId], TSDFDirection(currentHashEntry.direction),
//								pt_model, M_d, projParams_d, M_rgb, projParams_rgb,
//								this->settings->fusionParams, this->settings->sceneParams, depth, depthNormals,
//								depthImgSize, rgb, rgbImgSize);
						}
			}
		}
		this->timeStats.carving += timer.Tock();
	}

	/// 4. Collect per-voxel summation values, fuse into voxels
	timer.Tick();
	for (int entryId = 0; entryId < noVisibleEntries; entryId++)
	{
		const ITMHashEntry &currentHashEntry = hashTable[visibleEntryIds[entryId]];
		if (currentHashEntry.ptr < 0) continue;

		ITMVoxel *localVoxelBlock = &(localVBA[currentHashEntry.ptr * SDF_BLOCK_SIZE3]);
		const SummingVoxel *rayCastingSum = &(summingVoxelMap->getVoxels()[currentHashEntry.ptr * SDF_BLOCK_SIZE3]);

		for (int locId = 0; locId < SDF_BLOCK_SIZE3; locId++)
		{
//			rayCastCombine(localVoxelBlock[locId], rayCastingSum[locId], *(scene->sceneParams));
		}
	}
	this->timeStats.fusion += timer.Tock();
}

void ITMSceneReconstructionEngine_CPU::IntegrateIntoSceneVoxelProjection(
	Scene *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState)
{
	ITMTimer timer;
	timer.Tick();

	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, M_rgb;
	Vector4f projParams_d, projParams_rgb;

	ITMRenderState_VH *renderState_vh = (ITMRenderState_VH*)renderState;

	M_d = trackingState->pose_d->GetM();
	if (ITMVoxel::hasColorInformation) M_rgb = view->calib.trafo_rgb_to_depth.calib_inv * M_d;

	projParams_d = view->calib.intrinsics_d.projectionParamsSimple.all;
	projParams_rgb = view->calib.intrinsics_rgb.projectionParamsSimple.all;

	int maxW = scene->sceneParams->maxW;

	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
	Vector4f *depthNormals = nullptr;
	if (this->settings->fusionParams.useWeighting or this->settings->fusionParams.fusionMetric == FUSIONMETRIC_POINT_TO_PLANE)
		depthNormals = view->depthNormal->GetData(MEMORYDEVICE_CPU);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CPU);
	ITMVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	ITMHashEntry *hashTable = scene->index.GetEntries();

	int *visibleEntryIds = renderState_vh->GetVisibleEntryIDs();
	int noVisibleEntries = renderState_vh->noVisibleEntries;

	bool stopIntegratingAtMaxW = scene->sceneParams->stopIntegratingAtMaxW;
	//bool approximateIntegration = !trackingState->requiresFullRendering;

#ifdef WITH_OPENMP
	#pragma omp parallel for
#endif
	for (int entryId = 0; entryId < noVisibleEntries; entryId++)
	{
		const ITMHashEntry &currentHashEntry = hashTable[visibleEntryIds[entryId]];

		if (currentHashEntry.ptr < 0) continue;

		Vector3i globalPos = blockToVoxelPos(Vector3i(currentHashEntry.pos.x, currentHashEntry.pos.y, currentHashEntry.pos.z));

		ITMVoxel *localVoxelBlock = &(localVBA[currentHashEntry.ptr * (SDF_BLOCK_SIZE3)]);

		for (int z = 0; z < SDF_BLOCK_SIZE; z++) for (int y = 0; y < SDF_BLOCK_SIZE; y++) for (int x = 0; x < SDF_BLOCK_SIZE; x++)
		{
			Vector4f pt_model; int locId;

			locId = VoxelIndicesToOffset(x, y, z);

			if (stopIntegratingAtMaxW) if (localVoxelBlock[locId].w_depth == maxW) continue;
			//if (approximateIntegration) if (localVoxelBlock[locId].w_depth != 0) continue;

			pt_model.x = (float)(globalPos.x + x) * voxelSize;
			pt_model.y = (float)(globalPos.y + y) * voxelSize;
			pt_model.z = (float)(globalPos.z + z) * voxelSize;
			pt_model.w = 1.0f;

			std::conditional<ITMVoxel::hasColorInformation, ComputeUpdatedVoxelInfo<true, ITMVoxel>, ComputeUpdatedVoxelInfo<false, ITMVoxel>>::type::compute(
				localVoxelBlock[locId], TSDFDirection(currentHashEntry.direction), pt_model, M_d,
				projParams_d, M_rgb, projParams_rgb, this->settings->fusionParams, this->settings->sceneParams,
				depth, depthNormals, depthImgSize, rgb, rgbImgSize);
		}
	}
	this->timeStats.fusion += timer.Tock();
}

template<bool useSwapping>
void buildVisibleList_cpu(
	ITMHashEntry *hashTable,
	ITMHashSwapState *swapStates,
	const int noTotalEntries,
	int *visibleEntryIDs,
	AllocationStats *allocData,
	HashEntryVisibilityType *entriesVisibleType,
	const Matrix4f &M_d,
	const Vector4f &projParams_d,
	const Vector2i &depthImgSize,
	const float voxelSize
	)
{
//	for (int targetIdx = 0; targetIdx < noTotalEntries; targetIdx++)
//	{
//		buildVisibleList<useSwapping>(hashTable, swapStates, noTotalEntries, visibleEntryIDs,
//		                              allocData, entriesVisibleType, M_d, projParams_d,
//		                              depthImgSize, voxelSize, targetIdx);
//
//		if (entriesVisibleType[targetIdx] > 0)
//		{
//			visibleEntryIDs[allocData->noVisibleEntries] = targetIdx;
//			allocData->noVisibleEntries++;
//		}
//
//#if 0
//		// "active list", currently disabled
//		if (hashVisibleType == 1)
//		{
//			activeEntryIDs[noActiveEntries] = targetIdx;
//			noActiveEntries++;
//		}
//#endif
//	}
}

void ITMSceneReconstructionEngine_CPU::AllocateSceneFromDepth(Scene *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState, bool onlyUpdateVisibleList, bool resetVisibleList)
{
	ITMTimer timer;
	timer.Tick();
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, invM_d;
	Vector4f projParams_d, invProjParams_d;

	ITMRenderState_VH *renderState_vh = (ITMRenderState_VH*)renderState;
	if (resetVisibleList) renderState_vh->noVisibleEntries = 0;

	M_d = trackingState->pose_d->GetM(); M_d.inv(invM_d);

	projParams_d = view->calib.intrinsics_d.projectionParamsSimple.all;
	invProjParams_d = invertProjectionParams(projParams_d);

	float mu = scene->sceneParams->mu;

	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
	Vector4f *depthNormal = view->depthNormal->GetData(MEMORYDEVICE_CPU);
	int *voxelAllocationList = scene->localVBA.GetAllocationList();
	int *excessAllocationList = scene->index.GetExcessAllocationList();
	ITMHashEntry *hashTable = scene->index.GetEntries();
	ITMHashSwapState *swapStates = scene->globalCache != NULL ? scene->globalCache->GetSwapStates(false) : 0;
	int *visibleEntryIDs = renderState_vh->GetVisibleEntryIDs();
	HashEntryVisibilityType *entriesVisibleType = renderState_vh->GetEntriesVisibleType();
	HashEntryAllocType *entriesAllocType = this->entriesAllocType->GetData(MEMORYDEVICE_CPU);
	Vector4s *blockCoords = this->blockCoords->GetData(MEMORYDEVICE_CPU);
	TSDFDirection *blockDirections = this->blockDirections->GetData(MEMORYDEVICE_CPU);
	int noTotalEntries = scene->index.noTotalEntries;

	bool useSwapping = scene->globalCache != NULL;

	if (scene->localVBA.lastFreeBlockId <= 0)
	{
		printf("No more free blocks. Allocation stopped.\n");
	}

	memset(entriesAllocType, 0, noTotalEntries);

	for (int i = 0; i < renderState_vh->noVisibleEntries; i++)
	{
		entriesVisibleType[visibleEntryIDs[i]] = PREVIOUSLY_VISIBLE;
	}

	/// Compute visible and to-be-allocated blocks
#ifdef WITH_OPENMP
	#pragma omp parallel for
#endif
	for (int locId = 0; locId < depthImgSize.x*depthImgSize.y; locId++)
	{
		int y = locId / depthImgSize.x;
		int x = locId - y * depthImgSize.x;

		if (this->settings->fusionParams.useSpaceCarving)
			buildSpaceCarvingVisibleType(entriesVisibleType, x, y, blockCoords, blockDirections,
																	 depth, depthNormal, invM_d, invProjParams_d, mu, depthImgSize, voxelSize, hashTable,
																	 scene->sceneParams->viewFrustum_min, scene->sceneParams->viewFrustum_max,
																	 this->settings->fusionParams);

		buildHashAllocAndVisibleType(entriesAllocType, entriesVisibleType, x, y, blockCoords, blockDirections,
			depth, depthNormal, invM_d, invProjParams_d, mu, depthImgSize, voxelSize, hashTable,
			scene->sceneParams->viewFrustum_min, scene->sceneParams->viewFrustum_max,
			this->settings->fusionParams);
	}
	this->timeStats.buildingVisibilityList += timer.Tock();

	timer.Tick();
	if (onlyUpdateVisibleList) useSwapping = false;
	if (!onlyUpdateVisibleList)
	{
//		allocateVoxelBlocksList(voxelAllocationList, excessAllocationList, hashTable, noTotalEntries,
//			&allocationTempData, entriesAllocType, entriesVisibleType, blockCoords, blockDirections);
	}
	this->timeStats.allocation += timer.Tock();

	// Collect list of visible HashEntries
	timer.Tick();
	if (useSwapping)
	{
//		buildVisibleList_cpu<true>(hashTable, swapStates, noTotalEntries, visibleEntryIDs,
//															 &allocationTempData, entriesVisibleType, M_d, projParams_d,
//															 depthImgSize, voxelSize);
	} else{
//		buildVisibleList_cpu<false>(hashTable, swapStates, noTotalEntries, visibleEntryIDs,
//		                           &allocationTempData, entriesVisibleType, M_d, projParams_d,
//		                           depthImgSize, voxelSize);
	}
	this->timeStats.buildingVisibilityList += timer.Tock();

	//reallocate deleted ones from previous swap operation
	if (useSwapping)
	{
		timer.Tick();
		for (int targetIdx = 0; targetIdx < noTotalEntries; targetIdx++)
		{
			int vbaIdx;
			ITMHashEntry &hashEntry = hashTable[targetIdx];

			if (entriesVisibleType[targetIdx] > 0 && hashEntry.ptr == -1)
			{
//				vbaIdx = allocationTempData.noAllocatedVoxelEntries; allocationTempData.noAllocatedVoxelEntries--;
//				if (vbaIdx >= 0) hashEntry.ptr = voxelAllocationList[vbaIdx];
//				else allocationTempData.noAllocatedVoxelEntries++; // Avoid leaks
			}
		}
		this->timeStats.swapping += timer.Tock();
	}

//	renderState_vh->noVisibleEntries = allocationTempData.noVisibleEntries;
//
//	scene->localVBA.lastFreeBlockId = allocationTempData.noAllocatedVoxelEntries;
//	scene->index.SetLastFreeExcessListId(allocationTempData.noAllocatedExcessEntries);
//	memcpy(scene->localVBA.noAllocationsPerDirection,
//		     allocationTempData.noAllocationsPerDirection, sizeof(unsigned int) * N_DIRECTIONS);
}
void ITMSceneReconstructionEngine_CPU::FindVisibleBlocks(const Scene* scene, const ORUtils::SE3Pose* pose,
                                                          const ITMIntrinsics* intrinsics, ITMRenderState* renderState)
{
	printf("error: ITMSceneReconstructionEngine_CPU::FindVisibleBlocks not implemented\n");
//
//	const ITMHashEntry* hashTable = scene->index.GetEntries();
//	int noTotalEntries = scene->index.noTotalEntries;
//	float voxelSize = scene->sceneParams->voxelSize;
//	Vector2i imgSize = renderState->renderingRangeImage->noDims;
//
//	Matrix4f M = pose->GetM();
//	Vector4f projParams = intrinsics->projectionParamsSimple.all;
//
//	ITMRenderState_VH* renderState_vh = (ITMRenderState_VH*) renderState;
//
//	int noVisibleEntries = 0;
//	int* visibleEntryIDs = renderState_vh->GetVisibleEntryIDs();
//
//	//build visible list
//	for (int targetIdx = 0; targetIdx < noTotalEntries; targetIdx++)
//	{
//		const ITMHashEntry& hashEntry = hashTable[targetIdx];
//
//		if (hashEntry.ptr >= 0)
//		{
//			bool isVisible, isVisibleEnlarged;
//			checkBlockVisibility<false>(isVisible, isVisibleEnlarged, hashEntry.pos, M, projParams, voxelSize, imgSize);
//			visibleEntryIDs[noVisibleEntries] = targetIdx;
//			noVisibleEntries++;
//		}
//	}
//
//	renderState_vh->noVisibleEntries = noVisibleEntries;
}
