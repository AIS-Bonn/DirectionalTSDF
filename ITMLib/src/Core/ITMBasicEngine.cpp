// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include <Trackers/Shared/ITMICPTracker_Shared.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>

#include <cmath>
#include <ITMLib/Core/ITMBasicEngine.h>

#include <ITMLib/Engines/ITMLowLevelEngineFactory.h>
#include <ITMLib/Engines/ITMMeshingEngineFactory.h>
#include <ITMLib/Engines/ITMViewBuilderFactory.h>
#include <ITMLib/Engines/ITMVisualisationEngineFactory.h>
#include <ITMLib/Trackers/ITMTrackerFactory.h>
#include <Utils/ITMTimer.h>

#include <ORUtils/NVTimer.h>
#include <ORUtils/FileUtils.h>

using namespace ITMLib;

ITMBasicEngine::ITMBasicEngine(const std::shared_ptr<const ITMLibSettings>& settings, const ITMRGBDCalib& calib,
                               Vector2i imgSize_rgb, Vector2i imgSize_d)
	: settings(settings)
{
	if ((imgSize_d.x == -1) || (imgSize_d.y == -1)) imgSize_d = imgSize_rgb;

	MemoryDeviceType memoryType = settings->GetMemoryType();
	this->scene = new Scene(&settings->sceneParams, settings->swappingMode == ITMLibSettings::SWAPPINGMODE_ENABLED,
	                        settings->Directional(), memoryType);

	const ITMLibSettings::DeviceType deviceType = settings->deviceType;

	lowLevelEngine = ITMLowLevelEngineFactory::MakeLowLevelEngine(deviceType);
	viewBuilder = ITMViewBuilderFactory::MakeViewBuilder(calib, deviceType);
	visualisationEngine = ITMVisualisationEngineFactory::MakeVisualisationEngine(deviceType, settings);

	meshingEngine = nullptr;
	if (settings->createMeshingEngine)
		meshingEngine = ITMMeshingEngineFactory::MakeMeshingEngine(deviceType);

	denseMapper = new ITMDenseMapper(settings);
	denseMapper->ResetScene(scene);

	imuCalibrator = new ITMIMUCalibrator_iPad();
	tracker = ITMTrackerFactory::Instance().Make(imgSize_rgb, imgSize_d, settings, lowLevelEngine, imuCalibrator,
	                                             scene->sceneParams);
	trackingController = new ITMTrackingController(tracker, settings);

	Vector2i trackedImageSize = trackingController->GetTrackedImageSize(imgSize_rgb, imgSize_d);

	renderState_live = new ITMRenderState(trackedImageSize, scene->sceneParams->viewFrustum_min,
	                                      scene->sceneParams->viewFrustum_max, memoryType);
	renderState_freeview = nullptr; //will be created if needed

	trackingState = new ITMTrackingState(trackedImageSize, memoryType);
	tracker->UpdateInitialPose(trackingState);

	view = nullptr; // will be allocated by the view builder

	if (settings->behaviourOnFailure == settings->FAILUREMODE_RELOCALISE)
		relocaliser = new FernRelocLib::Relocaliser<float>(imgSize_d, Vector2f(settings->sceneParams.viewFrustum_min,
		                                                                       settings->sceneParams.viewFrustum_max), 0.2f,
		                                                   500, 4);
	else relocaliser = nullptr;

	kfRaycast = new ITMUChar4Image(imgSize_d, memoryType);

	trackingActive = true;
	fusionActive = true;
	mainProcessingActive = true;
	trackingInitialised = false;
	relocalisationCount = 0;
	framesProcessed = 0;
	consecutiveGoodFrames = 0;
}

ITMBasicEngine::~ITMBasicEngine()
{
	delete renderState_live;
	delete renderState_freeview;

	delete scene;

	delete denseMapper;
	delete trackingController;

	delete tracker;
	delete imuCalibrator;

	delete lowLevelEngine;
	delete viewBuilder;

	delete trackingState;
	delete view;

	delete visualisationEngine;

	delete relocaliser;
	delete kfRaycast;

	delete meshingEngine;
}

void ITMBasicEngine::SaveSceneToMesh(const char* objFileName)
{
	if (meshingEngine == nullptr) return;

	auto* mesh = new ITMMesh();

	meshingEngine->MeshScene(mesh, scene);
	mesh->WriteSTL(objFileName);

	delete mesh;
}

void ITMBasicEngine::SaveToFile()
{
	// throws error if any of the saves fail

	std::string saveOutputDirectory = "State/";
	std::string relocaliserOutputDirectory = saveOutputDirectory + "Relocaliser/";
	std::string sceneOutputDirectory = saveOutputDirectory + "Scene/";
	std::string renderOutputDirectory = saveOutputDirectory + "Render/";

	MakeDir(saveOutputDirectory.c_str());
	MakeDir(relocaliserOutputDirectory.c_str());
	MakeDir(sceneOutputDirectory.c_str());
	MakeDir(renderOutputDirectory.c_str());

	if (relocaliser) relocaliser->SaveToDirectory(relocaliserOutputDirectory);

	scene->SaveToDirectory(sceneOutputDirectory);
}

void ITMBasicEngine::LoadFromFile()
{
	std::string saveInputDirectory = "State/";
	std::string relocaliserInputDirectory = saveInputDirectory + "Relocaliser/", sceneInputDirectory =
		saveInputDirectory + "Scene/";

	////TODO: add factory for relocaliser and rebuild using config from relocaliserOutputDirectory + "config.txt"
	////TODO: add proper management of case when scene load fails (keep old scene or also reset relocaliser)

	this->resetAll();

	try // load relocaliser
	{
		auto* relocaliser_temp = new FernRelocLib::Relocaliser<float>(view->depth->noDims,
		                                                              Vector2f(settings->sceneParams.viewFrustum_min,
		                                                                       settings->sceneParams.viewFrustum_max), 0.2f,
		                                                              500, 4);

		relocaliser_temp->LoadFromDirectory(relocaliserInputDirectory);

		delete relocaliser;
		relocaliser = relocaliser_temp;
	}
	catch (std::runtime_error& e)
	{
		throw std::runtime_error("Could not load relocaliser: " + std::string(e.what()));
	}

	try // load scene
	{
		scene->LoadFromDirectory(sceneInputDirectory);
	}
	catch (std::runtime_error& e)
	{
		denseMapper->ResetScene(scene);
		throw std::runtime_error("Could not load scene:" + std::string(e.what()));
	}
}

void ITMBasicEngine::resetAll()
{
	denseMapper->ResetScene(scene);
	trackingState->Reset();
}

ITMTrackingState::TrackingResult
ITMBasicEngine::ProcessFrame(ITMUChar4Image* rgbImage, ITMShortImage* rawDepthImage, ITMIMUMeasurement* imuMeasurement,
                             const ORUtils::SE3Pose* pose)
{
	this->timeStats.Reset();
	ITMTimer timer;

	if (view)
	{
		if (rawDepthImage->noDims.width == 2000)
		{
			ITMRGBDCalib calibration;

			readRGBDCalib("/home/splietke/code/DirectionalTSDF/Files/COLMAP3.txt", calibration);
			view = new ITMView(calibration, rgbImage->noDims, rawDepthImage->noDims, true);
		}
	}

	bool computeNormals = true;
//	(
//		settings->fusionParams.useWeighting or
//		settings->fusionParams.tsdfMode == TSDFMode::TSDFMODE_DIRECTIONAL or
//		settings->fusionParams.fusionMode != FusionMode::FUSIONMODE_VOXEL_PROJECTION or
//		settings->fusionParams.fusionMetric == FusionMetric::FUSIONMETRIC_POINT_TO_PLANE);
	// prepare image and turn it into a depth image
	if (imuMeasurement == nullptr)
		viewBuilder->UpdateView(&view, rgbImage, rawDepthImage, settings->useDepthFilter, settings->useBilateralFilter, computeNormals);
	else
		viewBuilder->UpdateView(&view, rgbImage, rawDepthImage, settings->useDepthFilter, settings->useBilateralFilter, imuMeasurement,
		                        computeNormals);

	if (!mainProcessingActive) return ITMTrackingState::TRACKING_FAILED;

	// tracking
	ORUtils::SE3Pose oldPose(*(trackingState->pose_d));

	// If poses provided externally
	if (pose)
	{
		trackingState->trackerResult = ITMTrackingState::TRACKING_GOOD;
		trackingState->pose_d->SetFrom(pose);
		if (rawDepthImage->noDims.width == 2000)
		{
			Vector3f trans, rot;
			pose->GetParams(trans, rot);
			trackingState->pose_d->SetFrom(trans * (view->calib.disparityCalib.GetParams().x / 0.001), rot);
		}
	}
	else
	{
		// track pose (or refine, if pose given)
		if (trackingActive) trackingController->Track(trackingState, view);
	}

	// Rescale input according to computed scale factor
	lowLevelEngine->RescaleDepthImage(view->depth, std::exp(trackingState->scaleFactor));

	if (trackingState->trackerResult == ITMTrackingState::TRACKING_GOOD) consecutiveGoodFrames++;
	else consecutiveGoodFrames = 0;

	ITMTrackingState::TrackingResult trackerResult = ITMTrackingState::TRACKING_GOOD;
	switch (settings->behaviourOnFailure)
	{
		case ITMLibSettings::FAILUREMODE_RELOCALISE:
			trackerResult = trackingState->trackerResult;
			break;
		case ITMLibSettings::FAILUREMODE_STOP_INTEGRATION:
			if (trackingState->trackerResult != ITMTrackingState::TRACKING_FAILED)
				trackerResult = trackingState->trackerResult;
			else trackerResult = ITMTrackingState::TRACKING_POOR;
			break;
		default:
			break;
	}

	//relocalisation
	timer.Tick();
	int addKeyframeIdx = -1;
	if (settings->behaviourOnFailure == ITMLibSettings::FAILUREMODE_RELOCALISE)
	{
		if (trackerResult == ITMTrackingState::TRACKING_GOOD && relocalisationCount > 0) relocalisationCount--;

		int NN;
		float distances;
		view->depth->UpdateHostFromDevice();

		//find and add keyframe, if necessary
		bool hasAddedKeyframe = relocaliser->ProcessFrame(view->depth, trackingState->pose_d, 0, 1, &NN, &distances,
		                                                  trackerResult == ITMTrackingState::TRACKING_GOOD &&
		                                                  relocalisationCount == 0);

		//frame not added and tracking failed -> we need to relocalise
		if (!hasAddedKeyframe && trackerResult == ITMTrackingState::TRACKING_FAILED && trackingInitialised)
		{
			relocalisationCount = 10;

			const FernRelocLib::PoseDatabase::PoseInScene& keyframe = relocaliser->RetrievePose(NN);
			trackingState->pose_d->SetFrom(&keyframe.pose);

			denseMapper->UpdateVisibleList(view, trackingState, scene, renderState_live);
			trackingController->Prepare(trackingState, scene, view, visualisationEngine, renderState_live);
			trackingController->Track(trackingState, view);

			trackerResult = trackingState->trackerResult;

			addKeyframeIdx = 1;
		}
	}
	this->timeStats.relocalization.relocalization = timer.Tock();

	bool didFusion = false;
//	if ((trackerResult != ITMTrackingState::TRACKING_FAILED || !trackingInitialised)
	if ((trackerResult == ITMTrackingState::TRACKING_GOOD || !trackingInitialised)
	    && (fusionActive)
	    && (relocalisationCount == 0))
	{
		// fusion
		denseMapper->ProcessFrame(view, trackingState, scene, renderState_live);
		didFusion = true;
		if (consecutiveGoodFrames >= 10 or framesProcessed > 50) trackingInitialised = true;

		framesProcessed++;
	}

	if (trackerResult == ITMTrackingState::TRACKING_GOOD || trackerResult == ITMTrackingState::TRACKING_POOR ||
	    !trackingInitialised)
	{
		if (!didFusion) denseMapper->UpdateVisibleList(view, trackingState, scene, renderState_live);

		// raycast to renderState_live for tracking and free visualisation
		trackingController->Prepare(trackingState, scene, view, visualisationEngine, renderState_live);

		if (addKeyframeIdx >= 0)
		{
			ORUtils::MemoryCopyDirection memoryCopyDirection =
				settings->deviceType == ITMLibSettings::DEVICE_CUDA ? ORUtils::CUDA_TO_CUDA
				                                                    : ORUtils::CPU_TO_CPU;

			kfRaycast->SetFrom(renderState_live->renderedImage, memoryCopyDirection);
		}
	} else *trackingState->pose_d = oldPose;

	this->timeStats.preprocessing = viewBuilder->GetTimeStats();
	this->timeStats.tracking = trackingController->GetTimeStats();
//	this->timeStats.relocalization
	this->timeStats.reconstruction = denseMapper->GetSceneReconstructionEngine()->GetTimeStats();

	return trackerResult;
}

Vector2i ITMBasicEngine::GetImageSize() const
{
	return renderState_live->renderedImage->noDims;
}

template<typename T>
struct square : public thrust::unary_function<T, T>
{
	__host__ __device__ T operator()(const T& x) const
	{
		return x * x;
	}
};

ITMRenderError ITMBasicEngine::ComputeICPError()
{
	denseMapper->GetSceneReconstructionEngine()->FindVisibleBlocks(scene, trackingState->pose_d,
	                                                               &(view->calib.intrinsics_d), renderState_live);
	visualisationEngine->CreateExpectedDepths(scene, trackingState->pose_d, &(view->calib.intrinsics_d),
	                                          renderState_live);
	visualisationEngine->CreateICPMaps(scene, view, trackingState, renderState_live);
//	visualisationEngine->RenderTrackingError(renderState_live->renderedImage, trackingState, view);

	view->depth->UpdateHostFromDevice();
	ORUtils::Image<ORUtils::Vector4<float>> locations(trackingState->pointCloud->locations->noDims, true, false);
	ORUtils::Image<ORUtils::Vector4<float>> normals(trackingState->pointCloud->normals->noDims, true, false);

	ORcudaSafeCall(
		cudaMemcpy(locations.GetData(MEMORYDEVICE_CPU), trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CUDA),
		           locations.dataSize * sizeof(Vector4f), cudaMemcpyDeviceToHost));
	ORcudaSafeCall(
		cudaMemcpy(normals.GetData(MEMORYDEVICE_CPU), trackingState->pointCloud->normals->GetData(MEMORYDEVICE_CUDA),
		           normals.dataSize * sizeof(Vector4f), cudaMemcpyDeviceToHost));

	trackingState->pointCloud->locations->UpdateHostFromDevice();
	trackingState->pointCloud->normals->UpdateHostFromDevice();

	float* depth = view->depth->GetData(MEMORYDEVICE_CPU);

	const Vector4f* pointsRay = locations.GetData(MEMORYDEVICE_CPU);
	const Vector4f* normalsRay = normals.GetData(MEMORYDEVICE_CPU);
	const Matrix4f& depthImageInvPose = trackingState->pose_d->GetInvM();
	const Matrix4f& sceneRenderingPose = trackingState->pose_pointCloud->GetM();
	Vector2i imgSize = view->calib.intrinsics_d.imgSize;

	std::vector<float> errors, icpErrors;
	for (int x = 0; x < imgSize.width; x++)
		for (int y = 0; y < imgSize.height; y++)
		{
			float A[6];
			float error, icpError;
			float weight;
			bool isValidPoint = computePerPointGH_Depth_Ab<false, false>(
				A, icpError, weight, x, y,
				depthImageInvPose, sceneRenderingPose,
				depth,
				view->calib.intrinsics_d.imgSize, view->calib.intrinsics_d.projectionParamsSimple.all,
				view->calib.intrinsics_d.imgSize, view->calib.intrinsics_d.projectionParamsSimple.all,
				pointsRay, normalsRay, 100.0);

			isValidPoint &= computePerPointError<false, false>(
				error, x, y, depth,
				view->calib.intrinsics_d.imgSize, view->calib.intrinsics_d.projectionParamsSimple.all,
				view->calib.intrinsics_d.imgSize, view->calib.intrinsics_d.projectionParamsSimple.all,
				depthImageInvPose, sceneRenderingPose,
				pointsRay);

			if (!isValidPoint)
				continue;

			errors.push_back(std::fabs(error));
			icpErrors.push_back(std::fabs(icpError));
		}

	ITMRenderError result;
	result.MAE = thrust::reduce(errors.begin(), errors.end(), (float) 0, thrust::plus<float>()) / errors.size();
	result.RMSE = std::sqrt(thrust::transform_reduce(errors.begin(), errors.end(),
	                                                 square<float>(),
	                                                 (float) 0,
	                                                 thrust::plus<float>()) / errors.size());
	result.icpMAE =
		thrust::reduce(icpErrors.begin(), icpErrors.end(), (float) 0, thrust::plus<float>()) / icpErrors.size();
	result.icpRMSE = std::sqrt(thrust::transform_reduce(icpErrors.begin(), icpErrors.end(),
	                                                    square<float>(),
	                                                    (float) 0,
	                                                    thrust::plus<float>()) / icpErrors.size());

	return result;
}

void ITMBasicEngine::GetImage(ITMUChar4Image* out, const GetImageType getImageType, const ORUtils::SE3Pose* pose,
                              const ITMIntrinsics* intrinsics, bool normalsFromSDF)
{
	if (view == nullptr) return;

	out->Clear();

	IITMVisualisationEngine::RenderImageType renderImageType = ImageTypeToRenderType(getImageType, normalsFromSDF);

	switch (getImageType)
	{
		case ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_RGB:
			out->ChangeDims(view->rgb->noDims);
			if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
				out->SetFrom(view->rgb, ORUtils::CUDA_TO_CPU);
			else out->SetFrom(view->rgb, ORUtils::CPU_TO_CPU);
			break;
		case ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_DEPTH:
			out->ChangeDims(view->depth->noDims);
			if (settings->deviceType == ITMLibSettings::DEVICE_CUDA) view->depth->UpdateHostFromDevice();
			ITMVisualisationEngine::DepthToUchar4(out, view->depth);
			break;
		case ITMMainEngine::InfiniTAM_IMAGE_SCENERAYCAST:
		case ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_VOLUME:
		case ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_NORMAL:
		case ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_CONFIDENCE:
		case ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_DEPTH:
		{
			// use current raycast or forward projection?
			IITMVisualisationEngine::RenderRaycastSelection raycastType;
			if (trackingState->age_pointCloud <= 0) raycastType = IITMVisualisationEngine::RENDER_FROM_OLD_RAYCAST;
			else raycastType = IITMVisualisationEngine::RENDER_FROM_OLD_FORWARDPROJ;

			visualisationEngine->RenderImage(scene, trackingState->pose_d, &view->calib.intrinsics_d, renderState_live,
			                                 renderState_live->renderedImage, renderImageType, raycastType);

			ORUtils::Image<Vector4u>* srcImage;
			if (relocalisationCount != 0) srcImage = kfRaycast;
			else srcImage = renderState_live->renderedImage;

			out->ChangeDims(srcImage->noDims);
			if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
				out->SetFrom(srcImage, ORUtils::CUDA_TO_CPU);
			else out->SetFrom(srcImage, ORUtils::CPU_TO_CPU);

			break;
		}
		case ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_ICP_ERROR:
		{
			denseMapper->GetSceneReconstructionEngine()->FindVisibleBlocks(scene, trackingState->pose_d,
			                                                               &(view->calib.intrinsics_d), renderState_live);
			visualisationEngine->CreateExpectedDepths(scene, trackingState->pose_d, &(view->calib.intrinsics_d),
			                                          renderState_live);
			visualisationEngine->CreateICPMaps(scene, view, trackingState, renderState_live);
			visualisationEngine->RenderTrackingError(renderState_live->renderedImage, trackingState, view);
			out->ChangeDims(renderState_live->renderedImage->noDims);
			if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
				out->SetFrom(renderState_live->renderedImage, ORUtils::CUDA_TO_CPU);
			else out->SetFrom(renderState_live->renderedImage, ORUtils::CPU_TO_CPU);
			break;
		}
		case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_SHADED:
		case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME:
		case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL:
		case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_CONFIDENCE:
		case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_DEPTH:
		{
			if (renderState_freeview == nullptr)
			{
				renderState_freeview = new ITMRenderState(intrinsics->imgSize, scene->sceneParams->viewFrustum_min,
				                                          scene->sceneParams->viewFrustum_max, settings->GetMemoryType());
			}

			denseMapper->GetSceneReconstructionEngine()->FindVisibleBlocks(scene, pose, intrinsics,
			                                                               renderState_freeview);
			visualisationEngine->CreateExpectedDepths(scene, pose, intrinsics, renderState_freeview);
			visualisationEngine->RenderImage(scene, pose, intrinsics, renderState_freeview,
			                                 renderState_freeview->renderedImage, renderImageType,
			                                 IITMVisualisationEngine::RENDER_FROM_NEW_RAYCAST);

			if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
				out->SetFrom(renderState_freeview->renderedImage, ORUtils::CUDA_TO_CPU);
			else out->SetFrom(renderState_freeview->renderedImage, ORUtils::CPU_TO_CPU);
			break;
		}
		case ITMMainEngine::InfiniTAM_IMAGE_UNKNOWN:
			break;
	}
}

void ITMBasicEngine::turnOnIntegration()
{ fusionActive = true; }

void ITMBasicEngine::turnOffIntegration()
{ fusionActive = false; }

const unsigned int* ITMBasicEngine::GetAllocationsPerDirection()
{
	if (settings->Directional())
		return scene->tsdfDirectional->allocationStats.noAllocationsPerDirection;
	else
		return scene->tsdf->allocationStats.noAllocationsPerDirection;
}