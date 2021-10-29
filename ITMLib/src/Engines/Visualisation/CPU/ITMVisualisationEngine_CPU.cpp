// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMVisualisationEngine_CPU.h"
#include <unordered_map>
#include "../Shared/ITMVisualisationEngine_Shared.h"
#include <Engines/Reconstruction/Shared/ITMSceneReconstructionEngine_Shared.h>
#include <Objects/TSDF_CPU.h>

#include <vector>

using namespace ITMLib;

static int RenderPointCloud(Vector4f* locations, Vector4f* colours, const Vector4f* ptsRay,
                            const std::unordered_map<ITMIndex, ITMVoxel*>& tsdf,
                            bool skipPoints,
                            float oneOverVoxelSize,
                            Vector2i imgSize, Vector3f lightSource)
{
	int noTotalPoints = 0;

#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
	for (int y = 0, locId = 0; y < imgSize.y; y++)
		for (int x = 0; x < imgSize.x; x++, locId++)
		{
			Vector3f outNormal;
			float angle;
			const Vector4f& pointRay = ptsRay[locId];
			const Vector3f& point = pointRay.toVector3();
			bool foundPoint = pointRay.w > 0;

			computeNormalAndAngleTSDF(foundPoint, point, tsdf, oneOverVoxelSize, lightSource, outNormal, angle);

			if (skipPoints && ((x % 2 == 0) || (y % 2 == 0))) foundPoint = false;

			if (foundPoint)
			{
				colours[noTotalPoints] = readFromSDF_color4u_interpolated(tsdf, point);
				locations[noTotalPoints] = Vector4f(point, 1.0);

				noTotalPoints++;
			}
		}

	return noTotalPoints;
}

ITMRenderState* ITMVisualisationEngine_CPU::CreateRenderState(const Scene* scene,
                                                              const Vector2i& imgSize) const
{
	return new ITMRenderState(
		imgSize, scene->sceneParams->viewFrustum_min,
		scene->sceneParams->viewFrustum_max, MEMORYDEVICE_CPU
	);
}

void ITMVisualisationEngine_CPU::CreateExpectedDepths(const Scene* scene,
                                                      const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
                                                      ITMRenderState* renderState)
{
	ComputeRenderingTSDF(scene, pose, intrinsics, renderState);

	Vector2i imgSize = renderState->renderingRangeImage->noDims;
	Vector2f* minmaxData = renderState->renderingRangeImage->GetData(MEMORYDEVICE_CPU);

	std::fill_n(minmaxData, imgSize.x * imgSize.y, Vector2f(FAR_AWAY, VERY_CLOSE));

	float voxelSize = scene->sceneParams->voxelSize;

	std::vector<RenderingBlock> renderingBlocks(MAX_RENDERING_BLOCKS);
	int numRenderingBlocks = 0;

	int noVisibleEntries = renderState->noVisibleEntries;
	ITMIndex* visibleBlocks = renderState->GetVisibleBlocks();

	//go through list of visible 8x8x8 blocks
	for (int idx = 0; idx < noVisibleEntries; ++idx)
	{
		Vector3s blockIdx = visibleBlocks[idx].getPosition().toShort();

		Vector2i upperLeft, lowerRight;
		Vector2f zRange;
		bool validProjection;
		validProjection = ProjectSingleBlock(blockIdx, pose->GetM(), intrinsics->projectionParamsSimple.all, imgSize,
		                                     voxelSize, upperLeft, lowerRight, zRange);
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

ITMVisualisationEngine_CPU::ITMVisualisationEngine_CPU(const std::shared_ptr<const ITMLibSettings>& settings)
	: ITMVisualisationEngine(settings)
{
	if (settings->Directional())
	{
		this->renderingTSDF = new TSDF_CPU<ITMIndex, ITMVoxel>(settings->sceneParams.allocationSize / 4);
	}
}

void ITMVisualisationEngine_CPU::RenderImage(const Scene* scene,
                                             const ORUtils::SE3Pose* pose,
                                             const ITMIntrinsics* intrinsics,
                                             const ITMRenderState* renderState,
                                             ITMUChar4Image* outputImage,
                                             IITMVisualisationEngine::RenderImageType type,
                                             IITMVisualisationEngine::RenderRaycastSelection raycastType) const
{
	Vector2i imgSize = outputImage->noDims;
	Matrix4f invM = pose->GetInvM();

	Vector4f* pointsRay, * normalsRay;
	if (raycastType == IITMVisualisationEngine::RENDER_FROM_OLD_RAYCAST)
		pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CPU);
	else
	{
		if (raycastType == IITMVisualisationEngine::RENDER_FROM_OLD_FORWARDPROJ)
			pointsRay = renderState->forwardProjection->GetData(MEMORYDEVICE_CPU);
		else
		{
			// this one is generally done for freeview visualisation, so
			// no, do not update the list of visible blocks
			GenericRaycast(scene, imgSize, invM, intrinsics->projectionParamsSimple.all, renderState, false);
			pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CPU);
		}
	}
	normalsRay = renderState->raycastNormals->GetData(MEMORYDEVICE_CPU);

	Vector3f lightSource = Vector3f(invM.getColumn(3)) / scene->sceneParams->voxelSize;
	Vector4u* outRendering = outputImage->GetData(MEMORYDEVICE_CPU);

	TSDF_CPU<ITMIndex, ITMVoxel>* tsdf = GetRenderingTSDF(scene)->toCPU();

	switch (type)
	{
		case IITMVisualisationEngine::RENDER_COLOUR:
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
			for (int locId = 0; locId < imgSize.x * imgSize.y; locId++)
			{
				processPixelColour(outRendering[locId], pointsRay[locId],
				                   tsdf->getMap(), scene->sceneParams->oneOverVoxelSize, lightSource);
			}
			break;
		case IITMVisualisationEngine::RENDER_NORMAL_SDFNORMAL:
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
			for (int locId = 0; locId < imgSize.x * imgSize.y; locId++)
			{
				processPixelNormal_SDFNormals(outRendering[locId], pointsRay[locId], tsdf->getMap(),
				                              scene->sceneParams->oneOverVoxelSize, lightSource);
			}
			break;
		case IITMVisualisationEngine::RENDER_NORMAL_IMAGENORMAL:
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
			for (int y = 0; y < imgSize.y; y++)
				for (int x = 0; x < imgSize.x; x++)
				{
					processPixelNormals_ImageNormals<true>(outRendering, pointsRay, normalsRay, imgSize, x, y,
					                                       scene->sceneParams->voxelSize, lightSource);
				}
			break;
		case IITMVisualisationEngine::RENDER_CONFIDENCE_SDFNORMAL:
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
			for (int locId = 0; locId < imgSize.x * imgSize.y; locId++)
			{
				processPixelConfidence_SDFNormals(outRendering[locId], pointsRay[locId], tsdf->getMap(),
				                                  *(scene->sceneParams), lightSource);
			}
			break;
		case IITMVisualisationEngine::RENDER_CONFIDENCE_IMAGENORMAL:
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
			for (int y = 0; y < imgSize.y; y++)
				for (int x = 0; x < imgSize.x; x++)
				{
					processPixelConfidence_ImageNormals<true>(outRendering, pointsRay, normalsRay, imgSize, x, y,
					                                          *(scene->sceneParams), lightSource);
				}
			break;
		case IITMVisualisationEngine::RENDER_DEPTH_COLOUR:
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
			for (int locId = 0; locId < imgSize.x * imgSize.y; locId++)
			{
				processPixelDepthColour<ITMVoxel>(outRendering[locId], pointsRay[locId],
				                                  pose->GetM(), scene->sceneParams->viewFrustum_max);
			}
			break;
		case IITMVisualisationEngine::RENDER_DEPTH_IMAGENORMAL:
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
			for (int y = 0; y < imgSize.y; y++)
				for (int x = 0; x < imgSize.x; x++)
				{
					processPixelDepthShaded_ImageNormals<true>(outRendering, pointsRay, normalsRay, imgSize, x, y, lightSource);
				}
			break;
		case IITMVisualisationEngine::RENDER_DEPTH_SDFNORMAL:
		default:
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
			for (int locId = 0; locId < imgSize.x * imgSize.y; locId++)
			{
				processPixelDepthShaded_SDFNormals(outRendering[locId], pointsRay[locId],
				                                   tsdf->getMap(), scene->sceneParams->oneOverVoxelSize, lightSource);
			}
	}
}

void ITMVisualisationEngine_CPU::CreatePointCloud(const Scene* scene,
                                                  const ITMView* view,
                                                  ITMTrackingState* trackingState,
                                                  ITMRenderState* renderState, bool skipPoints) const
{
	Vector2i imgSize = renderState->raycastResult->noDims;
	Matrix4f invM = trackingState->pose_d->GetInvM() * view->calib.trafo_rgb_to_depth.calib;

	// this one is generally done for the colour tracker, so yes, update
	// the list of visible blocks if possible
	GenericRaycast(scene, imgSize, invM, view->calib.intrinsics_rgb.projectionParamsSimple.all, renderState, true);
	trackingState->pose_pointCloud->SetFrom(trackingState->pose_d);

	trackingState->pointCloud->noTotalPoints = RenderPointCloud(
		trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CPU),
		trackingState->pointCloud->colours->GetData(MEMORYDEVICE_CPU),
		renderState->raycastResult->GetData(MEMORYDEVICE_CPU),
		GetRenderingTSDF(scene)->toCPU()->getMap(),
		skipPoints,
		scene->sceneParams->oneOverVoxelSize,
		imgSize,
		Vector3f(invM.getColumn(3))
	);
}


void ITMVisualisationEngine_CPU::CreateICPMaps(const Scene* scene,
                                               const ITMView* view,
                                               ITMTrackingState* trackingState,
                                               ITMRenderState* renderState) const
{
	Vector2i imgSize = renderState->raycastResult->noDims;
	Matrix4f invM = trackingState->pose_d->GetInvM();

	// this one is generally done for the ICP tracker, so yes, update
	// the list of visible blocks if possible
	GenericRaycast(scene, imgSize, invM, view->calib.intrinsics_d.projectionParamsSimple.all, renderState, true);
	trackingState->pose_pointCloud->SetFrom(trackingState->pose_d);

	Vector3f lightSource = Vector3f(invM.getColumn(3)) / scene->sceneParams->voxelSize;
	Vector4f* normalsMap = trackingState->pointCloud->normals->GetData(MEMORYDEVICE_CPU);
	Vector4f* pointsMap = trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CPU);
	Vector4f* pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CPU);
	Vector4f* normalsRay = renderState->raycastNormals->GetData(MEMORYDEVICE_CPU);
	float voxelSize = scene->sceneParams->voxelSize;

#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
	for (int y = 0; y < imgSize.y; y++)
		for (int x = 0; x < imgSize.x; x++)
		{
			processPixelICP<true>(pointsMap, normalsMap, pointsRay, normalsRay, imgSize, x, y, voxelSize, lightSource);
		}
}

void ITMVisualisationEngine_CPU::ForwardRender(const Scene* scene,
                                               const ITMView* view,
                                               ITMTrackingState* trackingState,
                                               ITMRenderState* renderState) const
{
	Vector2i imgSize = renderState->raycastResult->noDims;
	Matrix4f M = trackingState->pose_d->GetM();
	Matrix4f invM = trackingState->pose_d->GetInvM();
	const Vector4f& projParams = view->calib.intrinsics_d.projectionParamsSimple.all;

	const Vector4f* pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CPU);
	Vector4f* forwardProjection = renderState->forwardProjection->GetData(MEMORYDEVICE_CPU);
	float* currentDepth = view->depth->GetData(MEMORYDEVICE_CPU);
	int* fwdProjMissingPoints = renderState->fwdProjMissingPoints->GetData(MEMORYDEVICE_CPU);
	const Vector2f* minmaximg = renderState->renderingRangeImage->GetData(MEMORYDEVICE_CPU);
	float voxelSize = scene->sceneParams->voxelSize;

	renderState->forwardProjection->Clear();

#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
	for (int y = 0; y < imgSize.y; y++)
		for (int x = 0; x < imgSize.x; x++)
		{
			int locId = x + y * imgSize.x;
			Vector4f pixel = pointsRay[locId];

			int locId_new = forwardProjectPixel(pixel * voxelSize, M, projParams, imgSize);
			if (locId_new >= 0) forwardProjection[locId_new] = pixel;
		}

	int noMissingPoints = 0;
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
	for (int y = 0; y < imgSize.y; y++)
		for (int x = 0; x < imgSize.x; x++)
		{
			int locId = x + y * imgSize.x;
			int locId2 =
				(int) floor((float) x / minmaximg_subsample) + (int) floor((float) y / minmaximg_subsample) * imgSize.x;

			Vector4f fwdPoint = forwardProjection[locId];
			Vector2f minmaxval = minmaximg[locId2];
			float depth = currentDepth[locId];

			if ((fwdPoint.w <= 0) && ((fwdPoint.x == 0 && fwdPoint.y == 0 && fwdPoint.z == 0) || (depth >= 0)) &&
			    (minmaxval.x < minmaxval.y))
				//if ((fwdPoint.w <= 0) && (minmaxval.x < minmaxval.y))
			{
				fwdProjMissingPoints[noMissingPoints] = locId;
				noMissingPoints++;
			}
		}

	renderState->noFwdProjMissingPoints = noMissingPoints;
	const Vector4f invProjParams = invertProjectionParams(projParams);

#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
	for (int pointId = 0; pointId < noMissingPoints; pointId++)
	{
		int locId = fwdProjMissingPoints[pointId];
		int y = locId / imgSize.x, x = locId - y * imgSize.x;
		int locId2 =
			(int) floor((float) x / minmaximg_subsample) + (int) floor((float) y / minmaximg_subsample) * imgSize.x;

		castRay(forwardProjection[locId], x, y,
		        scene->tsdf->toCPU()->getMap(),
		        invM, invProjParams, *(scene->sceneParams), minmaximg[locId2]);
	}
}

void ITMVisualisationEngine_CPU::GenericRaycast(const Scene* scene, const Vector2i& imgSize, const Matrix4f& invM,
                                                const Vector4f& projParams, const ITMRenderState* renderState,
                                                bool updateVisibleList) const
{
	Vector4f* pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CPU);
	const Vector2f* minmaximg = renderState->renderingRangeImage->GetData(MEMORYDEVICE_CPU);
	Vector4f invProjParams = invertProjectionParams(projParams);

	auto tsdf = GetRenderingTSDF(scene)->toCPU();

#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
	for (int y = 0; y < imgSize.y; y++)
		for (int x = 0; x < imgSize.x; x++)
		{
			int locId = x + y * imgSize.x;
			int locId2 =
				(int) floor((float) x / minmaximg_subsample) + (int) floor((float) y / minmaximg_subsample) * imgSize.x;

			float distance;
			castRayDefaultTSDF(pointsRay[locId], distance, x, y, tsdf->getMap(), invM, invProjParams,
			                   *(scene->sceneParams), minmaximg[locId2]);
		}

	Vector4f* normals = renderState->raycastNormals->GetData(MEMORYDEVICE_CPU);
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
	for (int y = 0; y < imgSize.y; y++)
		for (int x = 0; x < imgSize.x; x++)
		{
			int locId = x + y * imgSize.x;
			bool foundPoint = true;
			Vector3f normal;
			computeNormal<false>(pointsRay, scene->sceneParams->voxelSize, imgSize, x, y, foundPoint, normal);

			if (not foundPoint)
			{
				normals[locId] = Vector4f(0, 0, 0, -1);
				continue;
			}
			normals[locId] = Vector4f(normal, 1);
		}
}

void ITMVisualisationEngine_CPU::FindSurface(const Scene* scene,
                                             const ORUtils::SE3Pose* pose,
                                             const ITMIntrinsics* intrinsics,
                                             const ITMRenderState* renderState) const
{
	GenericRaycast(scene, renderState->raycastResult->noDims, pose->GetInvM(), intrinsics->projectionParamsSimple.all,
	               renderState, false);
}

void ITMVisualisationEngine_CPU::RenderTrackingError(ITMUChar4Image* outRendering,
                                                     const ITMTrackingState* trackingState,
                                                     const ITMView* view) const
{
	Vector4u* data = outRendering->GetData(MEMORYDEVICE_CPU);
	const Vector4f* pointsRay = trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CPU);
	const Vector4f* normalsRay = trackingState->pointCloud->normals->GetData(MEMORYDEVICE_CPU);
	const float* depthImage = view->depth->GetData(MEMORYDEVICE_CPU);
	const Matrix4f& depthImagePose = trackingState->pose_d->GetInvM();
	const Matrix4f& sceneRenderingPose = trackingState->pose_pointCloud->GetInvM();
	Vector2i imgSize = view->calib.intrinsics_d.imgSize;
	const float maxError = this->settings->sceneParams.mu;

	for (int y = 0; y < view->calib.intrinsics_d.imgSize.height; y++)
		for (int x = 0; x < view->calib.intrinsics_d.imgSize.width; x++)
		{
			processPixelError(data, pointsRay, normalsRay, depthImage, depthImagePose, sceneRenderingPose,
			                  view->calib.intrinsics_d.projectionParamsSimple.all, imgSize, maxError, x, y);
		}
}

void ITMVisualisationEngine_CPU::ComputeRenderingTSDFImpl(const Scene* scene, const ORUtils::SE3Pose* pose,
                                                          const ITMIntrinsics* intrinsics,
                                                          ITMRenderState* renderState)
{
	float voxelSize = scene->sceneParams->voxelSize;

	size_t N = renderState->noVisibleEntries;

	if (N == 0)
		return;

	auto renderingTSDF = this->renderingTSDF->toCPU();
	if (N >= renderingTSDF->allocatedBlocksMax)
		renderingTSDF->resize(2 * N);
	else
		renderingTSDF->clear();
	renderingTSDF->allocate(renderState->GetVisibleBlocks(), N);

	auto& renderingTSDF_map = renderingTSDF->getMap();
	auto& tsdf = scene->tsdfDirectional->toCPU()->getMap();
	const Matrix4f invM_d = pose->GetInvM();

	for (size_t i = 0; i < N; i++)
	{
		const ITMIndex& blockPos = renderState->GetVisibleBlocks()[i];
		auto it = renderingTSDF_map.find(blockPos);
		if (it == renderingTSDF_map.end())
			continue;
		ITMVoxel* block = it->second;

		Vector3i voxelPosIdx = blockToVoxelPos(Vector3i(blockPos.x, blockPos.y, blockPos.z));

#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
		for (int linearIdx = 0; linearIdx < SDF_BLOCK_SIZE3; linearIdx++)
		{
			ITMVoxel& voxel = block[linearIdx];
			Vector3f voxelPos = (voxelPosIdx + voxelOffsetToCoordinate(linearIdx)).toFloat();

			voxel = combineDirectionalTSDFViewPoint(voxelPos, tsdf, invM_d, voxelSize,
			                                        scene->sceneParams->mu, scene->sceneParams->maxW);
		}
	}
}
