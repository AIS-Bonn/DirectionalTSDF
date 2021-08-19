// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMMultiVisualisationEngine_CPU.h"

#include "../../../Objects/RenderStates/ITMRenderStateMultiScene.h"
#include "../../../Objects/Scene/ITMMultiSceneAccess.h"

#include "../Shared/ITMVisualisationEngine_Shared.h"

using namespace ITMLib;

ITMRenderState*
ITMMultiVisualisationEngine_CPU::CreateRenderState(const Scene* scene,
                                                   const Vector2i& imgSize) const
{
	return new ITMRenderStateMultiScene(imgSize, scene->sceneParams->viewFrustum_min,
	                                                             scene->sceneParams->viewFrustum_max, MEMORYDEVICE_CPU);
}

void ITMMultiVisualisationEngine_CPU::PrepareRenderState(
	const ITMVoxelMapGraphManager<ITMVoxel>& mapManager, ITMRenderState* _state)
{
	ITMRenderStateMultiScene* state = (ITMRenderStateMultiScene*) _state;

	state->PrepareLocalMaps(mapManager);
}

void ITMMultiVisualisationEngine_CPU::CreateExpectedDepths(const ORUtils::SE3Pose* pose,
                                                           const ITMIntrinsics* intrinsics,
                                                           ITMRenderState* _renderState) const
{
	ITMRenderStateMultiScene* renderState = (ITMRenderStateMultiScene*) _renderState;

	// reset min max image
	Vector2i imgSize = renderState->renderingRangeImage->noDims;
	Vector2f* minmaxData = renderState->renderingRangeImage->GetData(MEMORYDEVICE_CPU);

	for (int locId = 0; locId < imgSize.x * imgSize.y; ++locId)
	{
		Vector2f& pixel = minmaxData[locId];
		pixel.x = FAR_AWAY;
		pixel.y = VERY_CLOSE;
	}

	// add the values from each local map
	for (int localMapId = 0; localMapId < renderState->indexData_host.numLocalMaps; ++localMapId)
	{
		float voxelSize = renderState->sceneParams.voxelSize;
		const ITMHashEntry* hash_entries = renderState->indexData_host.index[localMapId];
		int noHashEntries = ITMVoxelBlockHash::noTotalEntries;

		std::vector<RenderingBlock> renderingBlocks(MAX_RENDERING_BLOCKS);
		int numRenderingBlocks = 0;

		Matrix4f localPose = pose->GetM() * renderState->indexData_host.posesInv[localMapId];
		for (int blockNo = 0; blockNo < noHashEntries; ++blockNo)
		{
			const ITMHashEntry& blockData(hash_entries[blockNo]);

			Vector2i upperLeft, lowerRight;
			Vector2f zRange;
			bool validProjection = false;
			if (blockData.ptr >= 0)
			{
				validProjection = ProjectSingleBlock(blockData.pos, localPose, intrinsics->projectionParamsSimple.all, imgSize,
				                                     voxelSize, upperLeft, lowerRight, zRange);
			}
			if (!validProjection) continue;

			Vector2i requiredRenderingBlocks(
				(int) ceilf((float) (lowerRight.x - upperLeft.x + 1) / (float) renderingBlockSizeX),
				(int) ceilf((float) (lowerRight.y - upperLeft.y + 1) / (float) renderingBlockSizeY));
			int requiredNumBlocks = requiredRenderingBlocks.x * requiredRenderingBlocks.y;

			if (numRenderingBlocks + requiredNumBlocks >= MAX_RENDERING_BLOCKS) continue;
			int offset = numRenderingBlocks;
			numRenderingBlocks += requiredNumBlocks;

			CreateRenderingBlocks(&(renderingBlocks[0]), offset, upperLeft, lowerRight, zRange);
		}

		// go through rendering blocks
		for (int blockNo = 0; blockNo < numRenderingBlocks; ++blockNo)
		{
			// fill minmaxData
			const RenderingBlock& b(renderingBlocks[blockNo]);

			for (int y = b.upperLeft.y; y <= b.lowerRight.y; ++y)
			{
				for (int x = b.upperLeft.x; x <= b.lowerRight.x; ++x)
				{
					Vector2f& pixel(minmaxData[x + y * imgSize.x]);
					if (pixel.x > b.zRange.x) pixel.x = b.zRange.x;
					if (pixel.y < b.zRange.y) pixel.y = b.zRange.y;
				}
			}
		}
	}
}

void ITMMultiVisualisationEngine_CPU::RenderImage(const ORUtils::SE3Pose* pose,
                                                  const ITMIntrinsics* intrinsics,
                                                  ITMRenderState* _renderState,
                                                  ITMUChar4Image* outputImage,
                                                  IITMVisualisationEngine::RenderImageType type) const
{
	ITMRenderStateMultiScene* renderState = (ITMRenderStateMultiScene*) _renderState;

	Vector2i imgSize = outputImage->noDims;
	Matrix4f invM = pose->GetInvM();

	bool useDirectioal = this->settings->fusionParams.tsdfMode == TSDFMode::TSDFMODE_DIRECTIONAL;

	Vector4f* pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CPU);
	Vector4f* normalsRay = renderState->raycastNormals->GetData(MEMORYDEVICE_CPU);
	Vector6f* directionalContribution = renderState->raycastDirectionalContribution->GetData(MEMORYDEVICE_CPU);

	// Generic Raycast
	float voxelSize = renderState->sceneParams.voxelSize;
	{
		Vector4f projParams = intrinsics->projectionParamsSimple.all;
		Vector4f invProjParams = invertProjectionParams(projParams);

		const Vector2f* minmaximg = renderState->renderingRangeImage->GetData(MEMORYDEVICE_CPU);

#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
		for (int locId = 0; locId < imgSize.x * imgSize.y; ++locId)
		{
			int y = locId / imgSize.x;
			int x = locId - y * imgSize.x;
			int locId2 =
				(int) floor((float) x / minmaximg_subsample) + (int) floor((float) y / minmaximg_subsample) * imgSize.x;

//			castRay<ITMMultiVoxel<ITMVoxel>, ITMMultiIndex<ITMVoxelIndex>>(pointsRay[locId], &directionalContribution[locId],
//			                                                               NULL, x, y, &renderState->voxelData_host,
//			                                                               &renderState->indexData_host, invM, invProjParams,
//			                                                               renderState->sceneParams, minmaximg[locId2],
//			                                                               useDirectioal);
		}

		Vector4f* normals = renderState->raycastNormals->GetData(MEMORYDEVICE_CPU);
		for (int locId = 0; locId < imgSize.x * imgSize.y; ++locId)
		{
			int y = locId / imgSize.x;
			int x = locId - y * imgSize.x;

			bool foundPoint = true;
			Vector3f normal;
			computeNormal<false, false>(pointsRay, voxelSize, imgSize, x, y, foundPoint, normal);

			if (not foundPoint)
			{
				normals[x + y * imgSize.x] = Vector4f(0, 0, 0, -1);
				continue;
			}

			normals[x + y * imgSize.x] = Vector4f(normal, 1);
		}
	}

	Vector3f lightSource = Vector3f(invM.getColumn(3)) / this->settings->sceneParams.voxelSize;
	Vector4u* outRendering = outputImage->GetData(MEMORYDEVICE_CPU);

	if ((type == IITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME) &&
	    (!ITMVoxel::hasColorInformation))
		type = IITMVisualisationEngine::RENDER_SHADED_GREYSCALE;

	switch (type)
	{
		case IITMVisualisationEngine::RENDER_COLOUR_FROM_VOLUME:
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
			for (int locId = 0; locId < imgSize.x * imgSize.y; locId++)
			{
				Vector4f ptRay = pointsRay[locId];
				processPixelColour<ITMMultiVoxel<ITMVoxel>, ITMMultiIndex<ITMVoxelIndex> >(
					outRendering[locId], ptRay.toVector3(),
					useDirectioal ? &directionalContribution[locId] : nullptr,
					ptRay.w > 0, &(renderState->voxelData_host),
					&(renderState->indexData_host), lightSource);
			}
			break;
		case IITMVisualisationEngine::RENDER_COLOUR_FROM_SDFNORMAL:
			printf("RENDER_COLOUR_FROM_SDFNORMAL not implemented\n");
			break;
		case IITMVisualisationEngine::RENDER_COLOUR_FROM_IMAGENORMAL:
			if (intrinsics->FocalLengthSignsDiffer())
			{
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
				for (int locId = 0; locId < imgSize.x * imgSize.y; locId++)
				{
					int y = locId / imgSize.x, x = locId - y * imgSize.x;
					processPixelNormals_ImageNormals<true, true>(outRendering, pointsRay, normalsRay, imgSize, x, y, voxelSize,
					                                             lightSource);
				}
			} else
			{
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
				for (int locId = 0; locId < imgSize.x * imgSize.y; locId++)
				{
					int y = locId / imgSize.x, x = locId - y * imgSize.x;
					processPixelNormals_ImageNormals<true, false>(outRendering, pointsRay, normalsRay, imgSize, x, y, voxelSize,
					                                              lightSource);
				}
			}
			break;
		case IITMVisualisationEngine::RENDER_COLOUR_FROM_CONFIDENCE_SDFNORMAL:
			printf("RENDER_COLOUR_FROM_CONFIDENCE_SDFNORMAL not implemented\n");
			break;
		case IITMVisualisationEngine::RENDER_COLOUR_FROM_CONFIDENCE_IMAGENORMAL:
			if (intrinsics->FocalLengthSignsDiffer())
			{
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
				for (int locId = 0; locId < imgSize.x * imgSize.y; locId++)
				{
					int y = locId / imgSize.x, x = locId - y * imgSize.x;
					processPixelConfidence_ImageNormals<true, true>(outRendering, pointsRay, normalsRay, imgSize, x, y,
					                                                renderState->sceneParams, lightSource);
				}
			} else
			{
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
				for (int locId = 0; locId < imgSize.x * imgSize.y; locId++)
				{
					int y = locId / imgSize.x, x = locId - y * imgSize.x;
					processPixelConfidence_ImageNormals<true, false>(outRendering, pointsRay, normalsRay, imgSize, x, y,
					                                                 renderState->sceneParams, lightSource);
				}
			}
			break;
		case IITMVisualisationEngine::RENDER_COLOUR_FROM_DEPTH:
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
			for (int locId = 0; locId < imgSize.x * imgSize.y; locId++)
			{
				processPixelDepth<ITMVoxel, ITMVoxelIndex>(outRendering[locId], pointsRay[locId].toVector3(),
				                                           pointsRay[locId].w > 0,
				                                           pose->GetM(), renderState->sceneParams.voxelSize,
				                                           renderState->sceneParams.viewFrustum_max);
			}
			break;
		case IITMVisualisationEngine::RENDER_SHADED_GREYSCALE:
		default:
			if (intrinsics->FocalLengthSignsDiffer())
			{
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
				for (int locId = 0; locId < imgSize.x * imgSize.y; locId++)
				{
					int y = locId / imgSize.x, x = locId - y * imgSize.x;
					processPixelGrey_ImageNormals<true, true>(outRendering, pointsRay, normalsRay, imgSize, x, y, voxelSize,
					                                          lightSource);
				}
			} else
			{
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
				for (int locId = 0; locId < imgSize.x * imgSize.y; locId++)
				{
					int y = locId / imgSize.x, x = locId - y * imgSize.x;
					processPixelGrey_ImageNormals<true, false>(outRendering, pointsRay, normalsRay, imgSize, x, y, voxelSize,
					                                           lightSource);
				}
			}
			break;
	}
}

