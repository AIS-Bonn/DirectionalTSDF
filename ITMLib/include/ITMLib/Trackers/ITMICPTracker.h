// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <vector>
#include <ITMLib/Trackers/ITMTracker.h>
#include <ITMLib/Engines/ITMLowLevelEngine.h>
#include <ITMLib/Objects/Tracking/ITMIntensityHierarchyLevel.h>
#include <ITMLib/Objects/Tracking/ITMImageHierarchy.h>
#include <ITMLib/Objects/Tracking/ITMTemplatedHierarchyLevel.h>
#include <ITMLib/Objects/Tracking/ITMSceneHierarchyLevel.h>
#include <ITMLib/Objects/Tracking/TrackerIterationType.h>

#include <Eigen/Dense>

#include <ORUtils/HomkerMap.h>
#include <ORUtils/SVMClassifier.h>
#include <sophus/se3.hpp>

namespace ITMLib
{

typedef double EigenT;

/** Base class for engine performing ICP based tracking.
		A typical example would be the original KinectFusion
		tracking algorithm.
*/
class ITMICPTracker : public ITMTracker
{
public:
	struct Parameters
	{
		enum class ColourMode
		{
			FRAME_TO_FRAME,
			FRAME_TO_KEYFRAME,
			FRAME_TO_RENDER,
			UNKNOWN
		};

		ColourMode ColourModeFromString(const std::string& s)
		{
			if (s == "f2f")
				return ColourMode::FRAME_TO_FRAME;
			else if (s == "f2kf")
				return ColourMode::FRAME_TO_KEYFRAME;
			else if (s == "f2r")
				return ColourMode::FRAME_TO_RENDER;
			return ColourMode::UNKNOWN;
		}

		std::vector<TrackerIterationType> levels;
		bool useDepth = true;
		bool useColour = false;
		float colourWeight = 0.1f;
		ColourMode colourMode = ColourMode::FRAME_TO_FRAME;
		bool optimizeScale = false;
		float minColourGradient = 0.01f;
		float smallStepSizeCriterion = 1e-3f;
		float outlierDistanceFine = 0.002f;
		float outlierDistanceCoarse = 0.01f;
		float outlierColourFine = 0.002f;
		float outlierColourCoarse = 0.01f;
		float failureDetectorThreshold = 3.0f;
		int numIterationsCoarse = 10;
		int numIterationsFine = 2;
	};

	void TrackCamera(ITMTrackingState* trackingState, const ITMView* view);

	bool requiresColourRendering() const
	{ return parameters.useColour && parameters.colourMode == Parameters::ColourMode::FRAME_TO_RENDER; }

	bool requiresDepthReliability() const
	{ return false; }

	bool requiresPointCloudRendering() const
	{ return true; }


	ITMICPTracker(Vector2i imgSize_d, Vector2i imgSize_rgb, const Parameters& parameters,
	              const ITMLowLevelEngine* lowLevelEngine, MemoryDeviceType memoryType);

	virtual ~ITMICPTracker(void);

private:
	const ITMLowLevelEngine* lowLevelEngine;

	ITMTrackingState* trackingState;
	const ITMView* view;

	int* noIterationsPerLevel;

	void TrackCameraSE3(ITMTrackingState* trackingState, const ITMView* view);

	void TrackCameraSim3(ITMTrackingState* trackingState, const ITMView* view);

	void SetupLevels();

	void PrepareForEvaluation();

	void SetEvaluationParams(int levelId);

	void ComputeDelta(Eigen::Matrix<EigenT, 6, 1>& delta, Eigen::Matrix<EigenT, 6, 1>& nabla,
	                  Eigen::Matrix<EigenT, 6, 6>& hessian, bool shortIteration) const;

	void ApplyDelta(const Eigen::Matrix<EigenT, 6, 1>& deltaSE3, Sophus::SE3<EigenT>& deltaT) const;

	bool HasConverged(const Eigen::Matrix<EigenT, 6, 1>& delta) const;

	void SetEvaluationData(ITMTrackingState* trackingState, const ITMView* view);

	void UpdatePoseQuality(int noValidPoints_old, const Eigen::Matrix<EigenT, 6, 6>& hessian_good, float f_old);

	ORUtils::HomkerMap* map;
	ORUtils::SVMClassifier* svmClassifier;
	Vector4f mu, sigma;
protected:
	Parameters parameters;

	float* distThresh;
	float* colourThresh;

	double lastNegativeEntropy = 0;
	double maxNegativeEntropy = 0;
	double averageNegativeEntropy = 0;
	double averageNegativeEntropy2 = 0;
	int averageNegativeEntropyCounter = 0;

	int levelId;
	TrackerIterationType iterationType;

	Matrix4f renderedScenePose;
	Matrix4f intensityReferencePose;

	/** Image hierarchy pyramid of point clouds rendered from scene */
	ITMImageHierarchy<ITMSceneHierarchyLevel>* sceneHierarchy = nullptr;
	/** Image hierarchy pyramid of input depth images */
	ITMImageHierarchy<ITMTemplatedHierarchyLevel<ITMFloatImage>>* viewHierarchy_depth = nullptr;
	/** Image hierarchy pyramid of input intensity images (from RGB) */
	ITMImageHierarchy<ITMIntensityHierarchyLevel>* viewHierarchy_intensity = nullptr;

	/** Image hierarchy pyramid of points reprojected from current depth image */
	ITMImageHierarchy<ITMTemplatedHierarchyLevel<ITMFloat4Image>>* reprojectedPointsHierarchy = nullptr;
	/** Image hierarchy pyramid of intensity values from the rendered image projected into the current intensity */
	ITMImageHierarchy<ITMTemplatedHierarchyLevel<ITMFloatImage>>* projectedIntensityHierarchy = nullptr;

	/** center of the point cloud rendered from scene */
	Vector3f scenePointCloudCenter_world;
	/** center of the point cloud from depth image */
	Vector3f depthPointCloudCenter;

	Matrix4f depthToRGBTransform;

	/**
	 * @param deltaT current estimate. Transform from current frame to rendered scene
	 */
	virtual int ComputeGandH_Depth(float& f, float* nabla, float* hessian, const Matrix4f& deltaT) = 0;

	/**
	 * @param deltaT current estimate. Transform from current frame to rendered scene
	 */
	virtual int ComputeGandH_RGB(float& f, float* nabla, float* hessian, const Matrix4f& deltaT) = 0;

	virtual void RenderRGBError(ITMUChar4Image* image_out, const Matrix4f& deltaT) = 0;

	virtual size_t ComputeGandHSim3_Depth(float& f, Eigen::Matrix<EigenT, 7, 7>& H, Eigen::Matrix<EigenT, 7, 1>& g,
	                                      const Matrix4f& approxInvPose) = 0;

	virtual size_t ComputeGandHSim3_RGB(float& f, Eigen::Matrix<EigenT, 7, 7>& H, Eigen::Matrix<EigenT, 7, 1>& g,
	                                    const Matrix4f& approxInvPose) = 0;

	virtual void ComputeDepthPointAndIntensity(ITMFloat4Image* points_out,
	                                           ITMFloatImage* intensity_out,
	                                           const ITMFloatImage* intensity_in,
	                                           const ITMFloatImage* depth_in,
	                                           const Vector4f& intrinsics_depth,
	                                           const Vector4f& intrinsics_rgb,
	                                           const Matrix4f& scenePose) = 0;
};
}
