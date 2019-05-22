// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMSceneReconstructionEngine_CPU.h"

#include "ITMLib/Objects/RenderStates/ITMRenderState_VH.h"
#include "ITMLib/Engines/Reconstruction/Shared/ITMFusionWeight.hpp"
#include "ITMLib/Engines/Reconstruction/Shared/ITMSceneReconstructionEngine_Shared.h"
using namespace ITMLib;

template<class TVoxel, class TIndex>
ITMSceneReconstructionEngine_CPU<TVoxel, TIndex>::ITMSceneReconstructionEngine_CPU(ITMLibSettings::TSDFMode tsdfMode,
			ITMLibSettings::FusionMode fusionMode, ITMLibSettings::FusionMetric fusionMetric)
	:ITMSceneReconstructionEngine<TVoxel, TIndex>(tsdfMode, fusionMode, fusionMetric)
{

}

template<class TVoxel>
ITMSceneReconstructionEngine_CPU<TVoxel,ITMVoxelBlockHash>::ITMSceneReconstructionEngine_CPU(ITMLibSettings::TSDFMode tsdfMode,
			ITMLibSettings::FusionMode fusionMode, ITMLibSettings::FusionMetric fusionMetric)
	: ITMSceneReconstructionEngine<TVoxel,ITMVoxelBlockHash>(tsdfMode, fusionMode, fusionMetric)
{
	int noTotalEntries = ITMVoxelBlockHash::noTotalEntries;
	entriesAllocType = new ORUtils::MemoryBlock<HashEntryAllocType>(noTotalEntries, MEMORYDEVICE_CPU);
	blockCoords = new ORUtils::MemoryBlock<Vector4s>(noTotalEntries, MEMORYDEVICE_CPU);
	blockDirections = new ORUtils::MemoryBlock<TSDFDirection>(noTotalEntries, MEMORYDEVICE_CPU);
}

template<class TVoxel>
ITMSceneReconstructionEngine_CPU<TVoxel,ITMVoxelBlockHash>::~ITMSceneReconstructionEngine_CPU(void)
{
	delete entriesAllocType;
	delete blockCoords;
	delete blockDirections;
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel,ITMVoxelBlockHash>::ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

	TVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
	for (int i = 0; i < numBlocks * blockSize; ++i) voxelBlocks_ptr[i] = TVoxel();
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

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMVoxelBlockHash>::IntegrateIntoScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState)
{
	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, M_rgb;
	Vector4f projParams_d, projParams_rgb;

	ITMRenderState_VH *renderState_vh = (ITMRenderState_VH*)renderState;

	M_d = trackingState->pose_d->GetM();
	if (TVoxel::hasColorInformation) M_rgb = view->calib.trafo_rgb_to_depth.calib_inv * M_d;

	projParams_d = view->calib.intrinsics_d.projectionParamsSimple.all;
	projParams_rgb = view->calib.intrinsics_rgb.projectionParamsSimple.all;

	int maxW = scene->sceneParams->maxW;

	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
	Vector4f *depthNormals = view->depthNormal->GetData(MEMORYDEVICE_CPU);
	float *confidence = view->depthConfidence->GetData(MEMORYDEVICE_CPU);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CPU);
	TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
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
		Vector3i globalPos;
		const ITMHashEntry &currentHashEntry = hashTable[visibleEntryIds[entryId]];

		if (currentHashEntry.ptr < 0) continue;

		globalPos.x = currentHashEntry.pos.x;
		globalPos.y = currentHashEntry.pos.y;
		globalPos.z = currentHashEntry.pos.z;
		globalPos *= SDF_BLOCK_SIZE;

		TVoxel *localVoxelBlock = &(localVBA[currentHashEntry.ptr * (SDF_BLOCK_SIZE3)]);

		for (int z = 0; z < SDF_BLOCK_SIZE; z++) for (int y = 0; y < SDF_BLOCK_SIZE; y++) for (int x = 0; x < SDF_BLOCK_SIZE; x++)
		{
			Vector4f pt_model; int locId;

			locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

			if (stopIntegratingAtMaxW) if (localVoxelBlock[locId].w_depth == maxW) continue;
			//if (approximateIntegration) if (localVoxelBlock[locId].w_depth != 0) continue;

			pt_model.x = (float)(globalPos.x + x) * voxelSize;
			pt_model.y = (float)(globalPos.y + y) * voxelSize;
			pt_model.z = (float)(globalPos.z + z) * voxelSize;
			pt_model.w = 1.0f;

			ComputeUpdatedVoxelInfo<TVoxel::hasColorInformation,TVoxel::hasConfidenceInformation, TVoxel>::compute(
				localVoxelBlock[locId], TSDFDirection(currentHashEntry.direction), pt_model, M_d,
				projParams_d, M_rgb, projParams_rgb, *(scene->sceneParams), depth, depthNormals, confidence, depthImgSize, rgb, rgbImgSize);
		}
	}
}

void allocateVoxelBlocksList(
	int *voxelAllocationList, int *excessAllocationList,
	ITMHashEntry *hashTable, int &noTotalEntries,
	AllocationTempData *allocationTempData,
	HashEntryAllocType *entriesAllocType, HashEntryVisibilityType *entriesVisibleType,
	Vector4s *blockCoords, TSDFDirection *blockDirections
	)
{
	for (int targetIdx = 0; targetIdx < noTotalEntries; targetIdx++)
	{
		int vbaIdx, exlIdx;

		switch (entriesAllocType[targetIdx])
		{
			case ALLOCATE_ORDERED: //needs allocation, fits in the ordered list
				vbaIdx = allocationTempData->noAllocatedVoxelEntries; allocationTempData->noAllocatedVoxelEntries--;

				if (vbaIdx >= 0) //there is room in the voxel block array
				{
					Vector4s pt_block_all = blockCoords[targetIdx];

					ITMHashEntry hashEntry;
					hashEntry.pos.x = pt_block_all.x; hashEntry.pos.y = pt_block_all.y; hashEntry.pos.z = pt_block_all.z;
					hashEntry.ptr = voxelAllocationList[vbaIdx];
					hashEntry.offset = 0;
					hashEntry.direction = static_cast<TSDFDirection_type>(blockDirections[targetIdx]);

					hashTable[targetIdx] = hashEntry;
				}
				else
				{
					// Mark entry as not visible since we couldn't allocate it but buildHashAllocAndVisibleType changed its state.
					entriesVisibleType[targetIdx] = INVISIBLE;

					// Restore previous value to avoid leaks.
					allocationTempData->noAllocatedVoxelEntries++;
				}

				break;
			case ALLOCATE_EXCESS: //needs allocation in the excess list
				vbaIdx = allocationTempData->noAllocatedVoxelEntries; allocationTempData->noAllocatedVoxelEntries--;
				exlIdx = allocationTempData->noAllocatedExcessEntries; allocationTempData->noAllocatedExcessEntries--;

				if (vbaIdx >= 0 && exlIdx >= 0) //there is room in the voxel block array and excess list
				{
					Vector4s pt_block_all = blockCoords[targetIdx];

					ITMHashEntry hashEntry;
					hashEntry.pos.x = pt_block_all.x; hashEntry.pos.y = pt_block_all.y; hashEntry.pos.z = pt_block_all.z;
					hashEntry.ptr = voxelAllocationList[vbaIdx];
					hashEntry.offset = 0;
					hashEntry.direction = static_cast<TSDFDirection_type>(blockDirections[targetIdx]);

					int exlOffset = excessAllocationList[exlIdx];

					hashTable[targetIdx].offset = exlOffset + 1; //connect to child

					hashTable[SDF_BUCKET_NUM + exlOffset] = hashEntry; //add child to the excess list

					entriesVisibleType[SDF_BUCKET_NUM + exlOffset] = VISIBLE_IN_MEMORY; //make child visible and in memory
				}
				else
				{
					// No need to mark the entry as not visible since buildHashAllocAndVisibleType did not mark it.
					// Restore previous value to avoid leaks.
					allocationTempData->noAllocatedVoxelEntries++;
					allocationTempData->noAllocatedExcessEntries++;
				}
			default:
				break;
		}
	}
}

template<bool useSwapping>
void buildVisibleList_cpu(
	ITMHashEntry *hashTable,
	ITMHashSwapState *swapStates,
	const int noTotalEntries,
	int *visibleEntryIDs,
	AllocationTempData *allocData,
	HashEntryVisibilityType *entriesVisibleType,
	const Matrix4f &M_d,
	const Vector4f &projParams_d,
	const Vector2i &depthImgSize,
	const float voxelSize
	)
{
	for (int targetIdx = 0; targetIdx < noTotalEntries; targetIdx++)
	{
		buildVisibleList<useSwapping>(hashTable, swapStates, noTotalEntries, visibleEntryIDs,
		                              allocData, entriesVisibleType, M_d, projParams_d,
		                              depthImgSize, voxelSize, targetIdx);

		if (entriesVisibleType[targetIdx] > 0)
		{
			visibleEntryIDs[allocData->noVisibleEntries] = targetIdx;
			allocData->noVisibleEntries++;
		}

#if 0
		// "active list", currently disabled
		if (hashVisibleType == 1)
		{
			activeEntryIDs[noActiveEntries] = targetIdx;
			noActiveEntries++;
		}
#endif
	}
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMVoxelBlockHash>::AllocateSceneFromDepth(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState, bool onlyUpdateVisibleList, bool resetVisibleList)
{
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, invM_d;
	Vector4f projParams_d, invProjParams_d;

	ITMRenderState_VH *renderState_vh = (ITMRenderState_VH*)renderState;
	if (resetVisibleList) renderState_vh->noVisibleEntries = 0;

	M_d = trackingState->pose_d->GetM(); M_d.inv(invM_d);

	projParams_d = view->calib.intrinsics_d.projectionParamsSimple.all;
	invProjParams_d = projParams_d;
	invProjParams_d.x = 1.0f / invProjParams_d.x;
	invProjParams_d.y = 1.0f / invProjParams_d.y;

	float mu = scene->sceneParams->mu;

	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
	Vector4f *depthNormal = nullptr;
	if (this->tsdfMode == ITMLibSettings::TSDFMode::TSDFMODE_DIRECTIONAL or
	    this->fusionMetric == ITMLibSettings::FusionMetric::FUSIONMETRIC_POINT_TO_PLANE)
		depthNormal = view->depthNormal->GetData(MEMORYDEVICE_CPU);
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

	float blockSideLength = voxelSize * SDF_BLOCK_SIZE;

	AllocationTempData allocationTempData;
	allocationTempData.noAllocatedVoxelEntries = scene->localVBA.lastFreeBlockId;
	allocationTempData.noAllocatedExcessEntries = scene->index.GetLastFreeExcessListId();
	allocationTempData.noVisibleEntries = 0;

	memset(entriesAllocType, 0, noTotalEntries);

	for (int i = 0; i < renderState_vh->noVisibleEntries; i++)
		entriesVisibleType[visibleEntryIDs[i]] = PREVIOUSLY_VISIBLE;

	/// Compute visible and to-be-allocated blocks
#ifdef WITH_OPENMP
	#pragma omp parallel for
#endif
	for (int locId = 0; locId < depthImgSize.x*depthImgSize.y; locId++)
	{
		int y = locId / depthImgSize.x;
		int x = locId - y * depthImgSize.x;
		buildHashAllocAndVisibleType(entriesAllocType, entriesVisibleType, x, y, blockCoords, blockDirections,
			depth, depthNormal, invM_d, invProjParams_d, mu, depthImgSize, blockSideLength, hashTable,
			scene->sceneParams->viewFrustum_min, scene->sceneParams->viewFrustum_max, this->tsdfMode, this->fusionMetric);
	}

	if (onlyUpdateVisibleList) useSwapping = false;
	if (!onlyUpdateVisibleList)
	{
		allocateVoxelBlocksList(voxelAllocationList, excessAllocationList, hashTable, noTotalEntries,
			&allocationTempData, entriesAllocType, entriesVisibleType, blockCoords, blockDirections);
	}

	// Collect list of visible HashEntries
	if (useSwapping)
	{
		buildVisibleList_cpu<true>(hashTable, swapStates, noTotalEntries, visibleEntryIDs,
															 &allocationTempData, entriesVisibleType, M_d, projParams_d,
															 depthImgSize, voxelSize);
	} else{
		buildVisibleList_cpu<false>(hashTable, swapStates, noTotalEntries, visibleEntryIDs,
		                           &allocationTempData, entriesVisibleType, M_d, projParams_d,
		                           depthImgSize, voxelSize);
	}

	//reallocate deleted ones from previous swap operation
	if (useSwapping)
	{
		for (int targetIdx = 0; targetIdx < noTotalEntries; targetIdx++)
		{
			int vbaIdx;
			ITMHashEntry &hashEntry = hashTable[targetIdx];

			if (entriesVisibleType[targetIdx] > 0 && hashEntry.ptr == -1)
			{
				vbaIdx = allocationTempData.noAllocatedVoxelEntries; allocationTempData.noAllocatedVoxelEntries--;
				if (vbaIdx >= 0) hashEntry.ptr = voxelAllocationList[vbaIdx];
				else allocationTempData.noAllocatedVoxelEntries++; // Avoid leaks
			}
		}
	}

	renderState_vh->noVisibleEntries = allocationTempData.noVisibleEntries;

	scene->localVBA.lastFreeBlockId = allocationTempData.noAllocatedVoxelEntries;
	scene->index.SetLastFreeExcessListId(allocationTempData.noAllocatedExcessEntries);
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel,ITMPlainVoxelArray>::ResetScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

	TVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
	for (int i = 0; i < numBlocks * blockSize; ++i) voxelBlocks_ptr[i] = TVoxel();
	int *vbaAllocationList_ptr = scene->localVBA.GetAllocationList();
	for (int i = 0; i < numBlocks; ++i) vbaAllocationList_ptr[i] = i;
	scene->localVBA.lastFreeBlockId = numBlocks - 1;
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMPlainVoxelArray>::AllocateSceneFromDepth(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState, bool onlyUpdateVisibleList, bool resetVisibleList)
{}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMPlainVoxelArray>::IntegrateIntoScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState)
{
	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, M_rgb;
	Vector4f projParams_d, projParams_rgb;

	M_d = trackingState->pose_d->GetM();
	if (TVoxel::hasColorInformation) M_rgb = view->calib.trafo_rgb_to_depth.calib_inv * M_d;

	projParams_d = view->calib.intrinsics_d.projectionParamsSimple.all;
	projParams_rgb = view->calib.intrinsics_rgb.projectionParamsSimple.all;

  int maxW = scene->sceneParams->maxW;

	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CPU);
	TVoxel *voxelArray = scene->localVBA.GetVoxelBlocks();

	const ITMPlainVoxelArray::IndexData *arrayInfo = scene->index.getIndexData();

	bool stopIntegratingAtMaxW = scene->sceneParams->stopIntegratingAtMaxW;
	//bool approximateIntegration = !trackingState->requiresFullRendering;

#ifdef WITH_OPENMP
	#pragma omp parallel for
#endif
	for (int locId = 0; locId < scene->index.getVolumeSize().x*scene->index.getVolumeSize().y*scene->index.getVolumeSize().z; ++locId)
	{
		int z = locId / (scene->index.getVolumeSize().x*scene->index.getVolumeSize().y);
		int tmp = locId - z * scene->index.getVolumeSize().x*scene->index.getVolumeSize().y;
		int y = tmp / scene->index.getVolumeSize().x;
		int x = tmp - y * scene->index.getVolumeSize().x;
		Vector4f pt_model;

		if (stopIntegratingAtMaxW) if (voxelArray[locId].w_depth == maxW) continue;
		//if (approximateIntegration) if (voxelArray[locId].w_depth != 0) continue;

		pt_model.x = (float)(x + arrayInfo->offset.x) * voxelSize;
		pt_model.y = (float)(y + arrayInfo->offset.y) * voxelSize;
		pt_model.z = (float)(z + arrayInfo->offset.z) * voxelSize;
		pt_model.w = 1.0f;

		ComputeUpdatedVoxelInfo<TVoxel::hasColorInformation, TVoxel::hasConfidenceInformation, TVoxel>::compute(
			voxelArray[locId], TSDFDirection::NONE, pt_model, M_d, projParams_d, M_rgb, projParams_rgb, scene->sceneParams,
			depth, depthImgSize, rgb, rgbImgSize);
	}
}
