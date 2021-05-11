// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include <Trackers/Shared/ITMDepthTracker_Shared.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include "ITMBasicEngine.h"

#include "ITMLib/Engines/LowLevel/ITMLowLevelEngineFactory.h"
#include "ITMLib/Engines/Meshing/ITMMeshingEngineFactory.h"
#include "ITMLib/Engines/ViewBuilding/ITMViewBuilderFactory.h"
#include "ITMLib/Engines/Visualisation/ITMVisualisationEngineFactory.h"
#include "ITMLib/Objects/RenderStates/ITMRenderStateFactory.h"
#include "ITMLib/Trackers/ITMTrackerFactory.h"
#include "ITMLib/Utils/ITMTimer.h"

#include "ORUtils/NVTimer.h"
#include "ORUtils/FileUtils.h"

//#define OUTPUT_TRAJECTORY_QUATERNIONS

using namespace ITMLib;

ITMBasicEngine::ITMBasicEngine(const std::shared_ptr<const ITMLibSettings>& settings, const ITMRGBDCalib& calib, Vector2i imgSize_rgb, Vector2i imgSize_d)
	:settings(settings)
{
	if ((imgSize_d.x == -1) || (imgSize_d.y == -1)) imgSize_d = imgSize_rgb;

	MemoryDeviceType memoryType = settings->GetMemoryType();
	this->scene = new Scene(&settings->sceneParams, settings->swappingMode == ITMLibSettings::SWAPPINGMODE_ENABLED, memoryType);

	const ITMLibSettings::DeviceType deviceType = settings->deviceType;

	lowLevelEngine = ITMLowLevelEngineFactory::MakeLowLevelEngine(deviceType);
	viewBuilder = ITMViewBuilderFactory::MakeViewBuilder(calib, deviceType);
	visualisationEngine = ITMVisualisationEngineFactory::MakeVisualisationEngine(deviceType, settings);

	meshingEngine = NULL;
	if (settings->createMeshingEngine)
		meshingEngine = ITMMeshingEngineFactory::MakeMeshingEngine(deviceType);

	denseMapper = new ITMDenseMapper(settings);
	denseMapper->ResetScene(scene);

	imuCalibrator = new ITMIMUCalibrator_iPad();
	tracker = ITMTrackerFactory::Instance().Make(imgSize_rgb, imgSize_d, settings, lowLevelEngine, imuCalibrator, scene->sceneParams);
	trackingController = new ITMTrackingController(tracker, settings);

	Vector2i trackedImageSize = trackingController->GetTrackedImageSize(imgSize_rgb, imgSize_d);

	renderState_live = ITMRenderStateFactory::CreateRenderState(trackedImageSize, scene->sceneParams, memoryType);
	renderState_freeview = NULL; //will be created if needed

	trackingState = new ITMTrackingState(trackedImageSize, memoryType);
	tracker->UpdateInitialPose(trackingState);

	view = NULL; // will be allocated by the view builder

	if (settings->behaviourOnFailure == settings->FAILUREMODE_RELOCALISE)
		relocaliser = new FernRelocLib::Relocaliser<float>(imgSize_d, Vector2f(settings->sceneParams.viewFrustum_min, settings->sceneParams.viewFrustum_max), 0.2f, 500, 4);
	else relocaliser = NULL;

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
	if (renderState_freeview != NULL) delete renderState_freeview;

	delete scene;

	delete denseMapper;
	delete trackingController;

	delete tracker;
	delete imuCalibrator;

	delete lowLevelEngine;
	delete viewBuilder;

	delete trackingState;
	if (view != NULL) delete view;

	delete visualisationEngine;

	if (relocaliser != NULL) delete relocaliser;
	delete kfRaycast;

	if (meshingEngine != NULL) delete meshingEngine;
}

void ITMBasicEngine::SaveSceneToMesh(const char *objFileName)
{
	if (meshingEngine == NULL) return;

	ITMMesh *mesh = new ITMMesh(settings->GetMemoryType());

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
	visualisationEngine->SaveRenderTSDF(renderOutputDirectory);
}

void ITMBasicEngine::LoadFromFile()
{
	std::string saveInputDirectory = "State/";
	std::string relocaliserInputDirectory = saveInputDirectory + "Relocaliser/", sceneInputDirectory = saveInputDirectory + "Scene/";

	////TODO: add factory for relocaliser and rebuild using config from relocaliserOutputDirectory + "config.txt"
	////TODO: add proper management of case when scene load fails (keep old scene or also reset relocaliser)

	this->resetAll();

	try // load relocaliser
	{
		FernRelocLib::Relocaliser<float> *relocaliser_temp = new FernRelocLib::Relocaliser<float>(view->depth->noDims, Vector2f(settings->sceneParams.viewFrustum_min, settings->sceneParams.viewFrustum_max), 0.2f, 500, 4);

		relocaliser_temp->LoadFromDirectory(relocaliserInputDirectory);

		delete relocaliser;
		relocaliser = relocaliser_temp;
	}
	catch (std::runtime_error &e)
	{
		throw std::runtime_error("Could not load relocaliser: " + std::string(e.what()));
	}

	try // load scene
	{
		scene->LoadFromDirectory(sceneInputDirectory);
	}
	catch (std::runtime_error &e)
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

#ifdef OUTPUT_TRAJECTORY_QUATERNIONS
static int QuaternionFromRotationMatrix_variant(const double *matrix)
{
	int variant = 0;
	if
		((matrix[4]>-matrix[8]) && (matrix[0]>-matrix[4]) && (matrix[0]>-matrix[8]))
	{
		variant = 0;
	}
	else if ((matrix[4]<-matrix[8]) && (matrix[0]>
		matrix[4]) && (matrix[0]> matrix[8])) {
		variant = 1;
	}
	else if ((matrix[4]> matrix[8]) && (matrix[0]<
		matrix[4]) && (matrix[0]<-matrix[8])) {
		variant = 2;
	}
	else if ((matrix[4]<
		matrix[8]) && (matrix[0]<-matrix[4]) && (matrix[0]< matrix[8])) {
		variant = 3;
	}
	return variant;
}

static void QuaternionFromRotationMatrix(const double *matrix, double *q) {
	/* taken from "James Diebel. Representing Attitude: Euler
	Angles, Quaternions, and Rotation Vectors. Technical Report, Stanford
	University, Palo Alto, CA."
	*/

	// choose the numerically best variant...
	int variant = QuaternionFromRotationMatrix_variant(matrix);
	double denom = 1.0;
	if (variant == 0) {
		denom += matrix[0] + matrix[4] + matrix[8];
	}
	else {
		int tmp = variant * 4;
		denom += matrix[tmp - 4];
		denom -= matrix[tmp % 12];
		denom -= matrix[(tmp + 4) % 12];
	}
	denom = sqrt(denom);
	q[variant] = 0.5*denom;

	denom *= 2.0;
	switch (variant) {
	case 0:
		q[1] = (matrix[5] - matrix[7]) / denom;
		q[2] = (matrix[6] - matrix[2]) / denom;
		q[3] = (matrix[1] - matrix[3]) / denom;
		break;
	case 1:
		q[0] = (matrix[5] - matrix[7]) / denom;
		q[2] = (matrix[1] + matrix[3]) / denom;
		q[3] = (matrix[6] + matrix[2]) / denom;
		break;
	case 2:
		q[0] = (matrix[6] - matrix[2]) / denom;
		q[1] = (matrix[1] + matrix[3]) / denom;
		q[3] = (matrix[5] + matrix[7]) / denom;
		break;
	case 3:
		q[0] = (matrix[1] - matrix[3]) / denom;
		q[1] = (matrix[6] + matrix[2]) / denom;
		q[2] = (matrix[5] + matrix[7]) / denom;
		break;
	}

	if (q[0] < 0.0f) for (int i = 0; i < 4; ++i) q[i] *= -1.0f;
}
#endif

ITMTrackingState::TrackingResult ITMBasicEngine::ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage, ITMIMUMeasurement *imuMeasurement, const ORUtils::SE3Pose* pose)
{
	this->timeStats.Reset();
	ITMTimer timer;

	bool modelSensorNoise = true;
//	(
//		settings->fusionParams.useWeighting or
//		settings->fusionParams.tsdfMode == TSDFMode::TSDFMODE_DIRECTIONAL or
//		settings->fusionParams.fusionMode != FusionMode::FUSIONMODE_VOXEL_PROJECTION or
//		settings->fusionParams.fusionMetric == FusionMetric::FUSIONMETRIC_POINT_TO_PLANE);
	// prepare image and turn it into a depth image
	if (imuMeasurement == NULL) viewBuilder->UpdateView(&view, rgbImage, rawDepthImage, settings->useBilateralFilter, modelSensorNoise);
	else viewBuilder->UpdateView(&view, rgbImage, rawDepthImage, settings->useBilateralFilter, imuMeasurement, modelSensorNoise);

	if (!mainProcessingActive) return ITMTrackingState::TRACKING_FAILED;

	// tracking
	ORUtils::SE3Pose oldPose(*(trackingState->pose_d));
	if (trackingActive) trackingController->Track(trackingState, view);

	// If poses provided externally
	if (pose)
	{
		trackingState->trackerResult = ITMTrackingState::TRACKING_GOOD;
		trackingState->pose_d->SetFrom(pose);
	}

	if (trackingState->trackerResult == ITMTrackingState::TRACKING_GOOD) consecutiveGoodFrames++;
	else consecutiveGoodFrames = 0;

	ITMTrackingState::TrackingResult trackerResult = ITMTrackingState::TRACKING_GOOD;
	switch (settings->behaviourOnFailure) {
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

		int NN; float distances;
		view->depth->UpdateHostFromDevice();

		//find and add keyframe, if necessary
		bool hasAddedKeyframe = relocaliser->ProcessFrame(view->depth, trackingState->pose_d, 0, 1, &NN, &distances, trackerResult == ITMTrackingState::TRACKING_GOOD && relocalisationCount == 0);

		//frame not added and tracking failed -> we need to relocalise
		if (!hasAddedKeyframe && trackerResult == ITMTrackingState::TRACKING_FAILED && trackingInitialised)
		{
			relocalisationCount = 10;

			// Reset previous rgb frame since the rgb image is likely different than the one acquired when setting the keyframe
			view->rgb_prev->Clear();

			const FernRelocLib::PoseDatabase::PoseInScene & keyframe = relocaliser->RetrievePose(NN);
			trackingState->pose_d->SetFrom(&keyframe.pose);

			denseMapper->UpdateVisibleList(view, trackingState, scene, renderState_live, true);
			trackingController->Prepare(trackingState, scene, view, visualisationEngine, renderState_live);
			trackingController->Track(trackingState, view);

			trackerResult = trackingState->trackerResult;

			addKeyframeIdx = 1;
		}
	}
	this->timeStats.relocalization.relocalization = timer.Tock();

	bool didFusion = false;
	if ((trackerResult != ITMTrackingState::TRACKING_FAILED || !trackingInitialised)
	    && (fusionActive)
	    && (relocalisationCount == 0)) {
		// fusion
		denseMapper->ProcessFrame(view, trackingState, scene, renderState_live);
		didFusion = true;
		if (consecutiveGoodFrames >= 10 or framesProcessed > 50) trackingInitialised = true;

		framesProcessed++;
	}

	if (trackerResult == ITMTrackingState::TRACKING_GOOD || trackerResult == ITMTrackingState::TRACKING_POOR || !trackingInitialised)
	{
		if (!didFusion) denseMapper->UpdateVisibleList(view, trackingState, scene, renderState_live);

		// raycast to renderState_live for tracking and free visualisation
		trackingController->Prepare(trackingState, scene, view, visualisationEngine, renderState_live);

		if (addKeyframeIdx >= 0)
		{
			ORUtils::MemoryBlock<Vector4u>::MemoryCopyDirection memoryCopyDirection =
				settings->deviceType == ITMLibSettings::DEVICE_CUDA ? ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CUDA : ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU;

			kfRaycast->SetFrom(renderState_live->raycastImage, memoryCopyDirection);
		}
	}
	else *trackingState->pose_d = oldPose;

#ifdef OUTPUT_TRAJECTORY_QUATERNIONS
	const ORUtils::SE3Pose *p = trackingState->pose_d;
	double t[3];
	double R[9];
	double q[4];
	for (int i = 0; i < 3; ++i) t[i] = p->GetInvM().m[3 * 4 + i];
	for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
		R[r * 3 + c] = p->GetM().m[c * 4 + r];
	QuaternionFromRotationMatrix(R, q);
	fprintf(stderr, "%f %f %f %f %f %f %f\n", t[0], t[1], t[2], q[1], q[2], q[3], q[0]);
#endif

	this->timeStats.preprocessing = viewBuilder->GetTimeStats();
	this->timeStats.tracking = trackingController->GetTimeStats();
//	this->timeStats.relocalization
	this->timeStats.reconstruction = denseMapper->GetSceneReconstructionEngine()->GetTimeStats();

  return trackerResult;
}

Vector2i ITMBasicEngine::GetImageSize(void) const
{
	return renderState_live->raycastImage->noDims;
}

template<typename T>
struct add_const_and_square : public thrust::unary_function<T,T>
{
	add_const_and_square(T v) : v(v) {}

	__host__ __device__ T operator()(const T &x) const
	{
		return (x + v) * (x + v);
	}

	T v;
};

ITMRenderError ITMBasicEngine::ComputeICPError()
{
	denseMapper->GetSceneReconstructionEngine()->FindVisibleBlocks(scene, trackingState->pose_d, &(view->calib.intrinsics_d), renderState_live);
	visualisationEngine->CreateExpectedDepths(scene, trackingState->pose_d, &(view->calib.intrinsics_d), renderState_live);
	visualisationEngine->CreateICPMaps(scene, view, trackingState, renderState_live);
//	visualisationEngine->RenderTrackingError(renderState_live->raycastImage, trackingState, view);

	view->depth->UpdateHostFromDevice();
	ORUtils::Image<ORUtils::Vector4<float>> locations(trackingState->pointCloud->locations->noDims, true, false);
	ORUtils::Image<ORUtils::Vector4<float>> colours(trackingState->pointCloud->locations->noDims, true, false);

	ORcudaSafeCall(cudaMemcpy(locations.GetData(MEMORYDEVICE_CPU), trackingState->pointCloud->locations->GetData(MEMORYDEVICE_CUDA),
													 locations.dataSize * sizeof(Vector4f), cudaMemcpyDeviceToHost));
	ORcudaSafeCall(cudaMemcpy(colours.GetData(MEMORYDEVICE_CPU), trackingState->pointCloud->colours->GetData(MEMORYDEVICE_CUDA),
	                          colours.dataSize * sizeof(Vector4f), cudaMemcpyDeviceToHost));

	trackingState->pointCloud->locations->UpdateHostFromDevice();
	trackingState->pointCloud->colours->UpdateHostFromDevice();

	float* depth = view->depth->GetData(MEMORYDEVICE_CPU);

	const Vector4f* pointsRay = locations.GetData(MEMORYDEVICE_CPU);
	const Vector4f* normalsRay = colours.GetData(MEMORYDEVICE_CPU);
	const float* depthImage = view->depth->GetData(MEMORYDEVICE_CUDA);
	const Matrix4f& depthImageInvPose = trackingState->pose_d->GetInvM();
	const Matrix4f& sceneRenderingPose = trackingState->pose_pointCloud->GetM();
	Vector2i imgSize = view->calib.intrinsics_d.imgSize;
	const float maxError = this->settings->sceneParams.mu;

	std::vector<float> errors;
	for (int x = 0; x < imgSize.width; x++)
		for (int y = 0; y < imgSize.height; y++)
		{

			int locId = x + y * imgSize.width;

			float d = depth[locId];
			const Vector4f& pt = pointsRay[locId];

			float A[6];
			float b;
			bool isValidPoint = computePerPointGH_Depth_Ab<false, false>(
				A, b, x, y, depth[locId],
				view->calib.intrinsics_d.imgSize, view->calib.intrinsics_d.projectionParamsSimple.all,
				view->calib.intrinsics_d.imgSize, view->calib.intrinsics_d.projectionParamsSimple.all,
			                                                             depthImageInvPose, sceneRenderingPose,
			                                                             pointsRay, normalsRay, 100.0);
			float angle = -(sceneRenderingPose * normalsRay[locId]).z;

			if (!isValidPoint)
				continue;

			b = fabs(b);
			errors.push_back(b);
		}

	ITMRenderError result;
	result.average = thrust::reduce(errors.begin(), errors.end(), (float) 0, thrust::plus<float>()) / errors.size();
	result.min = thrust::reduce(errors.begin(), errors.end(), (float) 0, thrust::minimum<float>());
	result.max = thrust::reduce(errors.begin(), errors.end(), (float) 0, thrust::maximum<float>());
	result.variance = thrust::transform_reduce(errors.begin(), errors.end(),
	                                           add_const_and_square<float>(-result.average),
	                                           (float) 0,
	                                           thrust::plus<float>()) / errors.size();

	return result;
}

void ITMBasicEngine::GetImage(ITMUChar4Image *out, GetImageType getImageType, ORUtils::SE3Pose *pose, const ITMIntrinsics *intrinsics, bool normalsFromSDF)
{
	if (view == NULL) return;

	out->Clear();

	IITMVisualisationEngine::RenderImageType renderImageType = ImageTypeToRenderType(getImageType, normalsFromSDF);

	switch (getImageType)
	{
	case ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_RGB:
		out->ChangeDims(view->rgb->noDims);
		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
			out->SetFrom(view->rgb, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
		else out->SetFrom(view->rgb, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
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
		IITMVisualisationEngine::RenderRaycastSelection raycastType = IITMVisualisationEngine::RENDER_FROM_NEW_RAYCAST;
		if (trackingState->age_pointCloud <= 0) raycastType = IITMVisualisationEngine::RENDER_FROM_OLD_RAYCAST;
		else raycastType = IITMVisualisationEngine::RENDER_FROM_OLD_FORWARDPROJ;

		visualisationEngine->RenderImage(scene, trackingState->pose_d, &view->calib.intrinsics_d, renderState_live, renderState_live->raycastImage, renderImageType, raycastType);

		ORUtils::Image<Vector4u> *srcImage = nullptr;
		if (relocalisationCount != 0) srcImage = kfRaycast;
		else srcImage = renderState_live->raycastImage;

		out->ChangeDims(srcImage->noDims);
		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
			out->SetFrom(srcImage, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
		else out->SetFrom(srcImage, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);

		break;
		}
	case ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_ICP_ERROR:
	{
		denseMapper->GetSceneReconstructionEngine()->FindVisibleBlocks(scene, trackingState->pose_d, &(view->calib.intrinsics_d), renderState_live);
		visualisationEngine->CreateExpectedDepths(scene, trackingState->pose_d, &(view->calib.intrinsics_d), renderState_live);
		visualisationEngine->CreateICPMaps(scene, view, trackingState, renderState_live);
		visualisationEngine->RenderTrackingError(renderState_live->raycastImage, trackingState, view);
		out->ChangeDims(renderState_live->raycastImage->noDims);
		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
			out->SetFrom(renderState_live->raycastImage, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
		else out->SetFrom(renderState_live->raycastImage, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		break;
	}
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_SHADED:
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME:
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL:
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_CONFIDENCE:
	case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_DEPTH:
	{
		if (renderState_freeview == NULL)
		{
			renderState_freeview = ITMRenderStateFactory::CreateRenderState(out->noDims, scene->sceneParams, settings->GetMemoryType());
		}

		denseMapper->GetSceneReconstructionEngine()->FindVisibleBlocks(scene, pose, &(view->calib.intrinsics_d), renderState_freeview);
		visualisationEngine->CreateExpectedDepths(scene, pose, intrinsics, renderState_freeview);
		visualisationEngine->RenderImage(scene, pose, intrinsics, renderState_freeview, renderState_freeview->raycastImage, renderImageType, IITMVisualisationEngine::RENDER_FROM_NEW_RAYCAST);

		if (settings->deviceType == ITMLibSettings::DEVICE_CUDA)
			out->SetFrom(renderState_freeview->raycastImage, ORUtils::MemoryBlock<Vector4u>::CUDA_TO_CPU);
		else out->SetFrom(renderState_freeview->raycastImage, ORUtils::MemoryBlock<Vector4u>::CPU_TO_CPU);
		break;
	}
	case ITMMainEngine::InfiniTAM_IMAGE_UNKNOWN:
		break;
	};
}

void ITMBasicEngine::turnOnTracking() { trackingActive = true; }

void ITMBasicEngine::turnOffTracking() { trackingActive = false; }

void ITMBasicEngine::turnOnIntegration() { fusionActive = true; }

void ITMBasicEngine::turnOffIntegration() { fusionActive = false; }

void ITMBasicEngine::turnOnMainProcessing() { mainProcessingActive = true; }

void ITMBasicEngine::turnOffMainProcessing() { mainProcessingActive = false; }
