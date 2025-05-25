// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <ITMLib/Core/ITMDenseMapper.h>
#include <ITMLib/Core/ITMMainEngine.h>
#include <ITMLib/Core/ITMTrackingController.h>
#include <ITMLib/Engines/ITMEvaluationEngine.h>
#include <ITMLib/Engines/ITMLowLevelEngine.h>
#include <ITMLib/Engines/ITMMeshingEngine.h>
#include <ITMLib/Engines/ITMViewBuilder.h>
#include <ITMLib/Engines/ITMVisualisationEngine.h>
#include <ITMLib/Objects/Misc/ITMIMUCalibrator.h>

#include <FernRelocLib/Relocaliser.h>

namespace ITMLib
{
class ITMBasicEngine : public ITMMainEngine
{
private:
	bool trackingActive, fusionActive, mainProcessingActive, trackingInitialised;
	int framesProcessed, consecutiveGoodFrames, relocalisationCount;

	ITMLowLevelEngine* lowLevelEngine;
	ITMVisualisationEngine* visualisationEngine;
	ITMEvaluationEngine* evaluationEngine;

	ITMMeshingEngine* meshingEngine;

	ITMViewBuilder* viewBuilder;
	ITMDenseMapper* denseMapper;
	ITMTrackingController* trackingController;

	Scene* scene;
	ITMRenderState* renderState_live;
	ITMRenderState* renderState_freeview;

	ITMTracker* tracker;
	ITMIMUCalibrator* imuCalibrator;

	FernRelocLib::Relocaliser<float>* relocaliser;
	ITMUChar4Image* kfRaycast;

	/// Pointer for storing the current input frame
	ITMView* view;

	/// Pointer to the current camera pose and additional tracking information
	ITMTrackingState* trackingState;

public:
	ITMView* GetView() override
	{ return view; }

	ITMTrackingState* GetTrackingState() override
	{ return trackingState; }

	ITMRenderState* GetRenderState() override
	{ return renderState_live; }

	ITMRenderState* GetRenderStateFreeview() override
	{ return renderState_freeview; }

	const unsigned int* GetAllocationsPerDirection() override;

	ITMRenderError ComputeICPError() override;

	ITMRenderError ComputePhotometricError() override;

	ITMTrackingState::TrackingResult
	ProcessFrame(ITMUChar4Image* rgbImage, ITMShortImage* rawDepthImage, ITMIMUMeasurement* imuMeasurement,
	             const ORUtils::SE3Pose* pose) override;

	/// Extracts a mesh from the current scene and saves it to the model file specified by the file name
	void SaveSceneToMesh(const char* fileName) override;

	/// save and load the full scene and relocaliser (if any) to/from file
	void SaveToFile() override;

	void LoadFromFile() override;

	/// Get a result image as output
	[[nodiscard]] Vector2i GetImageSize() const override;

	void GetImage(ITMUChar4Image* out, const GetImageType getImageType, const ORUtils::SE3Pose* pose,
	              const ITMIntrinsics* intrinsics, bool normalsFromSDF) override;

	void GetPointCloud(ITMPointCloud* out, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	                   bool normalsFromSDF) override;

	/// switch for turning integration on/off
	void turnOnIntegration();

	void turnOffIntegration();

	/// resets the scene and the tracker
	void resetAll();

	/** \brief Constructor
		Omitting a separate image size for the depth images
		will assume same resolution as for the RGB images.
	*/
	ITMBasicEngine(const std::shared_ptr<const ITMLibSettings>& settings, const ITMRGBDCalib& calib, Vector2i imgSize_rgb,
	               Vector2i imgSize_d = Vector2i(-1, -1));

	~ITMBasicEngine() override;
};
}
