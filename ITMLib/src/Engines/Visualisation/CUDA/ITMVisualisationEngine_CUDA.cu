// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMVisualisationEngine_CUDA.h"
#include "ITMVisualisationHelpers_CUDA.h"

#include <stdgpu/unordered_set.cuh>
#include <stdgpu/unordered_map.cuh>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <Objects/TSDF_CUDA.h>
#include <Utils/ITMTimer.h>
#include <Utils/ITMBlockTraversal.h>
#include <ORUtils/FileUtils.h>

namespace ITMLib
{

inline dim3 getGridSize(dim3 taskSize, dim3 blockSize)
{
	return dim3((taskSize.x + blockSize.x - 1) / blockSize.x, (taskSize.y + blockSize.y - 1) / blockSize.y,
	            (taskSize.z + blockSize.z - 1) / blockSize.z);
}

inline dim3 getGridSize(Vector2i taskSize, dim3 blockSize)
{ return getGridSize(dim3(taskSize.x, taskSize.y), blockSize); }

ITMVisualisationEngine_CUDA::ITMVisualisationEngine_CUDA(std::shared_ptr<const ITMLibSettings> settings)
	: ITMVisualisationEngine(settings)
{
	ORcudaSafeCall(cudaMalloc((void**) &renderingBlockList_device, sizeof(RenderingBlock) * MAX_RENDERING_BLOCKS));
	ORcudaSafeCall(cudaMalloc((void**) &noTotalBlocks_device, sizeof(uint)));
	ORcudaSafeCall(cudaMalloc((void**) &this->noTotalPoints_device, sizeof(uint)));
	ORcudaSafeCall(cudaMalloc((void**) &noVisibleEntries_device, sizeof(uint)));

	if (settings->Directional())
	{
		this->renderingTSDF = new TSDF_CUDA<ITMIndex, ITMVoxel>(settings->sceneParams.allocationSize / 4);
	}
}

ITMVisualisationEngine_CUDA::~ITMVisualisationEngine_CUDA()
{
	ORcudaSafeCall(cudaFree(this->noTotalPoints_device));
	ORcudaSafeCall(cudaFree(noTotalBlocks_device));
	ORcudaSafeCall(cudaFree(renderingBlockList_device));
	ORcudaSafeCall(cudaFree(noVisibleEntries_device));
}

ITMRenderState* ITMVisualisationEngine_CUDA::CreateRenderState(const Scene* scene, const Vector2i& imgSize) const
{
	return new ITMRenderState(
		imgSize, scene->sceneParams->viewFrustum_min,
		scene->sceneParams->viewFrustum_max, MEMORYDEVICE_CUDA
	);
}

void ITMVisualisationEngine_CUDA::ComputeRenderingTSDFImpl(const Scene* scene, const ORUtils::SE3Pose* pose,
                                                           const ITMLib::ITMIntrinsics* intrinsics,
                                                           ITMRenderState* renderState)
{
	Matrix4f M = pose->GetM();

	size_t N = renderState->noVisibleEntries;

	if (N == 0)
		return;

	auto renderingTSDF_device = this->renderingTSDF->toCUDA();
	if (N >= renderingTSDF->allocatedBlocksMax)
		renderingTSDF->resize(2 * N);
	else
		renderingTSDF->clear();
	renderingTSDF_device->allocate(renderState->GetVisibleBlocks(), N);

	auto tsdf = scene->tsdfDirectional->toCUDA();
	dim3 gridSize(N, 1);
	dim3 blockSize(8 * 8 * 8);
//	combineDirectionalTSDFViewPoint_opt_device << < gridSize, blockSize >> > (
	combineDirectionalTSDFViewPoint_device << < gridSize, blockSize >> > (
		renderingTSDF_device->getMap(),
			tsdf->getMap(), renderState->GetVisibleBlocks(), N,
			pose->GetInvM(), scene->sceneParams->voxelSize, scene->sceneParams->mu, scene->sceneParams->maxW);
	ORcudaKernelCheck;
}

void ITMVisualisationEngine_CUDA::CreateExpectedDepths(const Scene* scene, const ORUtils::SE3Pose* pose,
                                                       const ITMIntrinsics* intrinsics,
                                                       ITMRenderState* renderState)
{
	ComputeRenderingTSDF(scene, pose, intrinsics, renderState);

	float voxelSize = scene->sceneParams->voxelSize;

	Vector2i imgSize = renderState->renderingRangeImage->noDims;
	Vector2f* minmaxData = renderState->renderingRangeImage->GetData(MEMORYDEVICE_CUDA);

	Vector2f init;
	init.x = FAR_AWAY;
	init.y = VERY_CLOSE;
	memsetKernel<Vector2f>(minmaxData, init, renderState->renderingRangeImage->dataSize);

	//go through list of visible 8x8x8 blocks, project to image plane, create list of 16x16 rendering blocks
	{
		int noVisibleEntries = renderState->noVisibleEntries;
		if (noVisibleEntries == 0) return;

		dim3 blockSize(256);
		dim3 gridSize((int) ceil((float) noVisibleEntries / (float) blockSize.x));
		ORcudaSafeCall(cudaMemset(noTotalBlocks_device, 0, sizeof(uint)));
		projectAndSplitBlocks_device << < gridSize, blockSize >> > (
			renderingBlockList_device, noTotalBlocks_device, renderState->GetVisibleBlocks(),
				noVisibleEntries, pose->GetM(), intrinsics->projectionParamsSimple.all, imgSize, voxelSize);
		ORcudaKernelCheck;
	}

	uint noTotalBlocks;
	ORcudaSafeCall(cudaMemcpy(&noTotalBlocks, noTotalBlocks_device, sizeof(uint), cudaMemcpyDeviceToHost));
	if (noTotalBlocks == 0) return;
	if (noTotalBlocks > (unsigned) MAX_RENDERING_BLOCKS) noTotalBlocks = MAX_RENDERING_BLOCKS;

	// go through rendering blocks, compute per pixel min/max z value for faster raycasting
	{
		// fill minmaxData
		dim3 blockSize(16, 16);
		dim3 gridSize((unsigned int) ceil((float) noTotalBlocks / 4.0f), 4);
		computeMinMaxData_device << <
		gridSize, blockSize >> > (noTotalBlocks, renderingBlockList_device, imgSize, minmaxData);
		ORcudaKernelCheck;
	}
}

void ITMVisualisationEngine_CUDA::GenericRaycast(const Scene* scene, const Vector2i& imgSize, const Matrix4f& invM,
                                                 const Vector4f& projParams, const ITMRenderState* renderState,
                                                 bool updateVisibleList) const
{
	dim3 cudaBlockSize(8, 8);
	dim3 gridSize((int) ceil((float) imgSize.x / (float) cudaBlockSize.x),
	              (int) ceil((float) imgSize.y / (float) cudaBlockSize.y));


	auto tsdf = GetRenderingTSDF(scene)->toCUDA();

	genericRaycast_device << < gridSize, cudaBlockSize >> > (
		renderState->raycastResult->GetData(MEMORYDEVICE_CUDA),
			tsdf->getMap(),
			imgSize,
			invM,
			invertProjectionParams(projParams),
			*(scene->sceneParams),
			renderState->renderingRangeImage->GetData(MEMORYDEVICE_CUDA)
	);
	ORcudaKernelCheck;

	if (settings->useSDFNormals)
	{
		computeSDFNormals_device << < gridSize, cudaBlockSize >> > (
			renderState->raycastNormals->GetData(MEMORYDEVICE_CUDA),
				renderState->raycastResult->GetData(MEMORYDEVICE_CUDA),
				GetRenderingTSDF(scene)->toCUDA()->getMap(),
				scene->sceneParams->oneOverVoxelSize, imgSize, Vector3f(invM.getColumn(3)));
		ORcudaKernelCheck;
	} else
	{
		computePointCloudNormals_device<ITMVoxel> << < gridSize, cudaBlockSize >> > (
			renderState->raycastNormals->GetData(MEMORYDEVICE_CUDA),
				renderState->raycastResult->GetData(MEMORYDEVICE_CUDA),
				imgSize, scene->sceneParams->voxelSize);
		ORcudaKernelCheck;
	}
}

void ITMVisualisationEngine_CUDA::RenderImage(const Scene* scene,
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
	{
		pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);
	} else if (raycastType == IITMVisualisationEngine::RENDER_FROM_OLD_FORWARDPROJ)
	{
		pointsRay = renderState->forwardProjection->GetData(MEMORYDEVICE_CUDA);
	} else
	{
		GenericRaycast(scene, imgSize, invM, intrinsics->projectionParamsSimple.all, renderState, false);
		pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);
	}
	normalsRay = renderState->raycastNormals->GetData(MEMORYDEVICE_CUDA);

	Vector3f lightSource = Vector3f(invM.getColumn(3));
	Vector4u* outRendering = outputImage->GetData(MEMORYDEVICE_CUDA);

	auto tsdf = GetRenderingTSDF(scene)->toCUDA();

	dim3 cudaBlockSize(8, 8);
	dim3 gridSize((int) ceil((float) imgSize.x / (float) cudaBlockSize.x),
	              (int) ceil((float) imgSize.y / (float) cudaBlockSize.y));

	switch (type)
	{
		case IITMVisualisationEngine::RENDER_COLOUR:
			renderColour_device << < gridSize, cudaBlockSize >> > (outRendering, pointsRay,
		tsdf->getMap(), scene->sceneParams->oneOverVoxelSize,
		imgSize, lightSource);
			ORcudaKernelCheck;
			break;
		case IITMVisualisationEngine::RENDER_NORMAL_SDFNORMAL:
			renderNormals_device << < gridSize, cudaBlockSize >> >
			(outRendering, pointsRay, tsdf->getMap(), imgSize, scene->sceneParams->oneOverVoxelSize, lightSource);
			ORcudaKernelCheck;
			break;
		case IITMVisualisationEngine::RENDER_NORMAL_IMAGENORMAL:
			renderNormals_ImageNormals_device << < gridSize, cudaBlockSize >> >
			(outRendering, pointsRay, normalsRay, imgSize, scene->sceneParams->voxelSize, lightSource);
			ORcudaKernelCheck;
			break;
		case IITMVisualisationEngine::RENDER_CONFIDENCE_SDFNORMAL:
			renderConfidence_device << < gridSize, cudaBlockSize >> > (outRendering, pointsRay,
		tsdf->getMap(), imgSize, *(scene->sceneParams), lightSource);
			ORcudaKernelCheck;
			break;
		case IITMVisualisationEngine::RENDER_CONFIDENCE_IMAGENORMAL:
			renderConfidence_ImageNormals_device << < gridSize, cudaBlockSize >> >
			(outRendering, pointsRay, normalsRay, imgSize, *(scene->sceneParams), lightSource);
			ORcudaKernelCheck;
			break;
		case IITMVisualisationEngine::RENDER_DEPTH_COLOUR:
			renderDepthColour_device<ITMVoxel> << < gridSize, cudaBlockSize >> > (outRendering, pointsRay, pose->GetM(),
		imgSize, scene->sceneParams->viewFrustum_max);
			ORcudaKernelCheck;
			break;
		case IITMVisualisationEngine::RENDER_DEPTH_IMAGENORMAL:
			renderDepthShaded_ImageNormals_device << < gridSize, cudaBlockSize >> >
			(outRendering, pointsRay, normalsRay, imgSize, lightSource);
			ORcudaKernelCheck;
			break;
		case IITMVisualisationEngine::RENDER_DEPTH_SDFNORMAL:
		default:
			renderDepthShaded_device << < gridSize, cudaBlockSize >> >
			(outRendering, pointsRay, tsdf->getMap(), scene->sceneParams->oneOverVoxelSize, imgSize, lightSource);
			ORcudaKernelCheck;
			break;
	}
}

void ITMVisualisationEngine_CUDA::ForwardRender(const Scene* scene,
                                                const ITMView* view,
                                                ITMTrackingState* trackingState,
                                                ITMRenderState* renderState) const
{
	Vector2i imgSize = renderState->raycastResult->noDims;
	Matrix4f M = trackingState->pose_d->GetM();
	Matrix4f invM = trackingState->pose_d->GetInvM();
	const Vector4f& projParams = view->calib.intrinsics_d.projectionParamsSimple.all;

	const Vector4f* pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);
	float* currentDepth = view->depth->GetData(MEMORYDEVICE_CUDA);
	Vector4f* forwardProjection = renderState->forwardProjection->GetData(MEMORYDEVICE_CUDA);
	int* fwdProjMissingPoints = renderState->fwdProjMissingPoints->GetData(MEMORYDEVICE_CUDA);
	const Vector2f* minmaximg = renderState->renderingRangeImage->GetData(MEMORYDEVICE_CUDA);
	float voxelSize = scene->sceneParams->voxelSize;

	renderState->forwardProjection->Clear();

	dim3 blockSize, gridSize;

	{ // forward projection
		blockSize = dim3(16, 16);
		gridSize = dim3((int) ceil((float) imgSize.x / (float) blockSize.x),
		                (int) ceil((float) imgSize.y / (float) blockSize.y));

		forwardProject_device << < gridSize, blockSize >> >
		(forwardProjection, pointsRay, imgSize, M, projParams, voxelSize);
		ORcudaKernelCheck;
	}

	ORcudaSafeCall(cudaMemset(noTotalPoints_device, 0, sizeof(uint)));

	{ // find missing points
		blockSize = dim3(16, 16);
		gridSize = dim3((int) ceil((float) imgSize.x / (float) blockSize.x),
		                (int) ceil((float) imgSize.y / (float) blockSize.y));

		findMissingPoints_device << < gridSize, blockSize >> > (fwdProjMissingPoints, noTotalPoints_device, minmaximg,
			forwardProjection, currentDepth, imgSize);
		ORcudaKernelCheck;
	}

	ORcudaSafeCall(
		cudaMemcpy(&renderState->noFwdProjMissingPoints, noTotalPoints_device, sizeof(uint), cudaMemcpyDeviceToHost));

	{ // render missing points
		blockSize = dim3(256);
		gridSize = dim3((int) ceil((float) renderState->noFwdProjMissingPoints / blockSize.x));

		std::cerr << "genericRaycastMissingPoints_device not implemented" << std::endl;
	}
}

void ITMVisualisationEngine_CUDA::CreatePointCloud(const Scene* scene,
                                                   const ITMView* view,
                                                   ITMTrackingState* trackingState,
                                                   ITMRenderState* renderState,
                                                   bool skipPoints) const
{
	Vector2i imgSize = renderState->raycastResult->noDims;
	Matrix4f invM = trackingState->pose_d->GetInvM() * view->calib.trafo_rgb_to_depth.calib;

	GenericRaycast(scene, imgSize, invM, view->calib.intrinsics_rgb.projectionParamsSimple.all, renderState, true);
	trackingState->pose_pointCloud->SetFrom(trackingState->pose_d);

	ORcudaSafeCall(cudaMemsetAsync(noTotalPoints_device, 0, sizeof(uint)));

	Vector3f lightSource = Vector3f(invM.getColumn(3));
	Vector4f* locations = trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CUDA);
	Vector4f* colours = trackingState->pointCloud->colours->GetData(MEMORYDEVICE_CUDA);
	Vector4f* pointsRay = renderState->raycastResult->GetData(MEMORYDEVICE_CUDA);

	dim3 cudaBlockSize(8, 8);
	dim3 gridSize = getGridSize(imgSize, cudaBlockSize);

	renderPointCloud_device << < gridSize, cudaBlockSize >> > (locations, colours, noTotalPoints_device, pointsRay,
		renderingTSDF->toCUDA()->getMap(),
		skipPoints, scene->sceneParams->voxelSize, imgSize, lightSource);
	ORcudaKernelCheck;

	ORcudaSafeCall(
		cudaMemcpy(&trackingState->pointCloud->noTotalPoints, noTotalPoints_device, sizeof(uint), cudaMemcpyDeviceToHost));
}

void ITMVisualisationEngine_CUDA::FindSurface(const Scene* scene,
                                              const ORUtils::SE3Pose* pose,
                                              const ITMIntrinsics* intrinsics,
                                              const ITMRenderState* renderState) const
{
	GenericRaycast(scene, renderState->raycastResult->noDims, pose->GetInvM(), intrinsics->projectionParamsSimple.all,
	               renderState, false);
}

void ITMVisualisationEngine_CUDA::CreateICPMaps(const Scene* scene,
                                                const ITMView* view,
                                                ITMTrackingState* trackingState,
                                                ITMRenderState* renderState) const
{
	Vector2i imgSize = renderState->raycastResult->noDims;
	Matrix4f invM = trackingState->pose_d->GetInvM();

	GenericRaycast(scene, imgSize, invM, view->calib.intrinsics_d.projectionParamsSimple.all, renderState, true);
	trackingState->pose_pointCloud->SetFrom(trackingState->pose_d);
	trackingState->pointCloud->locations->SetFrom(renderState->raycastResult, ORUtils::CUDA_TO_CUDA);
	trackingState->pointCloud->normals->SetFrom(renderState->raycastNormals, ORUtils::CUDA_TO_CUDA);
}

void ITMVisualisationEngine_CUDA::RenderTrackingError(ITMUChar4Image* outRendering,
                                                      const ITMTrackingState* trackingState,
                                                      const ITMView* view) const
{
	Vector4u* data = outRendering->GetData(MEMORYDEVICE_CUDA);
	const Vector4f* pointsRay = trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CUDA);
	const Vector4f* normalsRay = trackingState->pointCloud->normals->GetData(MEMORYDEVICE_CUDA);
	const float* depthImage = view->depth->GetData(MEMORYDEVICE_CUDA);
	const Matrix4f& depthImageInvPose = trackingState->pose_d->GetInvM();
	const Matrix4f& sceneRenderingPose = trackingState->pose_pointCloud->GetM();
	Vector2i imgSize = view->calib.intrinsics_d.imgSize;
	const float maxError = this->settings->sceneParams.mu;

	dim3 cudaBlockSize(8, 8);
	dim3 gridSize((int) ceil((float) imgSize.x / (float) cudaBlockSize.x),
	              (int) ceil((float) imgSize.y / (float) cudaBlockSize.y));
	renderPixelError_device<ITMVoxel> << < gridSize, cudaBlockSize >> > (
		data, pointsRay, normalsRay, depthImage, depthImageInvPose, sceneRenderingPose,
			view->calib.intrinsics_d.projectionParamsSimple.all, imgSize, maxError);
}

} // namespace ITMLib
