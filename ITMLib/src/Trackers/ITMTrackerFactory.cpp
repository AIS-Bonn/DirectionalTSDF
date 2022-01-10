//
// Created by Malte Splietker on 28.10.21.
//

#include <ITMLib/Trackers/ITMTrackerFactory.h>

#include "CPU/ITMColorTracker_CPU.h"
#include "CPU/ITMICPTracker_CPU.h"
#include "CPU/ITMExtendedTracker_CPU.h"

#ifndef COMPILE_WITHOUT_CUDA

#include "CUDA/ITMColorTracker_CUDA.h"
#include "CUDA/ITMICPTracker_CUDA.h"
#include "CUDA/ITMExtendedTracker_CUDA.h"

#endif

namespace ITMLib {

ITMTracker* ITMTrackerFactory::Make(ITMLibSettings::DeviceType deviceType, const char* trackerConfig, const Vector2i& imgSize_rgb,
                 const Vector2i& imgSize_d, const ITMLowLevelEngine* lowLevelEngine,
                 ITMIMUCalibrator* imuCalibrator, const ITMSceneParams* sceneParams) const
{
	ORUtils::KeyValueConfig cfg(trackerConfig);
	int verbose = 0;
	if (cfg.getProperty("help") != nullptr) if (verbose < 10) verbose = 10;

	ORUtils::KeyValueConfig::ChoiceList trackerOptions;
	for (int i = 0; (unsigned) i < makers.size(); ++i)
	{
		trackerOptions.addChoice(makers[i].id, makers[i].type);
	}
	int type = TRACKER_ICP;
	cfg.parseChoiceProperty("type", "type of tracker", type, trackerOptions, verbose);
	const Maker* maker = nullptr;
	for (int i = 0; (unsigned) i < makers.size(); ++i)
	{
		if (makers[i].type == type)
		{
			maker = &(makers[i]);
			break;
		}
	}
	if (maker == nullptr) DIEWITHEXCEPTION("Unknown tracker type");

	ITMTracker* ret = (*(maker->make))(imgSize_rgb, imgSize_d, deviceType, cfg, lowLevelEngine, imuCalibrator,
	                                   sceneParams);
	if (ret->requiresColourRendering())
	{
		printf("Assuming a voxel type with colour information!");
	}

	return ret;
}

/**
 * \brief Makes a tracker of the type specified in the settings.
 */
ITMTracker*
ITMTrackerFactory::Make(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, const std::shared_ptr<const ITMLibSettings>& settings,
     const ITMLowLevelEngine* lowLevelEngine,
     ITMIMUCalibrator* imuCalibrator, const ITMSceneParams* sceneParams) const
{
	return Make(settings->deviceType, settings->trackerConfig.c_str(), imgSize_rgb, imgSize_d, lowLevelEngine,
	            imuCalibrator, sceneParams);
}

//#################### PUBLIC STATIC MEMBER FUNCTIONS ####################
std::vector<TrackerIterationType> ITMTrackerFactory::parseLevelConfig(const char* str)
{
	bool parseError = false;
	std::vector<TrackerIterationType> ret;
	for (int i = static_cast<int>(strlen(str)) - 1; i >= 0; --i)
	{
		switch (str[i])
		{
			case 'r':
				ret.push_back(TRACKER_ITERATION_ROTATION);
				break;
			case 't':
				ret.push_back(TRACKER_ITERATION_TRANSLATION);
				break;
			case 'b':
				ret.push_back(TRACKER_ITERATION_BOTH);
				break;
			case 'n':
				ret.push_back(TRACKER_ITERATION_NONE);
				break;
			default:
				parseError = true;
				break;
		}
	}

	if (parseError)
	{
		fprintf(stderr, "error parsing level configuration '%s'\n", str);
		for (int i = 0; (unsigned) i < ret.size(); ++i)
			fprintf(stderr, "level %i: %i\n", (int) ret.size() - i, (int) (ret[ret.size() - i]));
	}
	return ret;
}

/**
 * \brief Makes a colour tracker.
 */
ITMTracker*
ITMTrackerFactory::MakeColourTracker(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, ITMLibSettings::DeviceType deviceType,
                  const ORUtils::KeyValueConfig& cfg,
                  const ITMLowLevelEngine* lowLevelEngine, ITMIMUCalibrator* imuCalibrator,
                  const ITMSceneParams* sceneParams)
{
	int verbose = 0;
	if (cfg.getProperty("help") != nullptr) if (verbose < 10) verbose = 10;

	const char* levelSetup = "rrrbb";
	cfg.parseStrProperty("levels", "resolution hierarchy levels", levelSetup, verbose);
	std::vector<TrackerIterationType> levels = parseLevelConfig(levelSetup);

	ITMColorTracker* ret = nullptr;
	switch (deviceType)
	{
		case ITMLibSettings::DEVICE_CPU:
			ret = new ITMColorTracker_CPU(imgSize_rgb, &(levels[0]), static_cast<int>(levels.size()), lowLevelEngine);
			break;
		case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
			ret = new ITMColorTracker_CUDA(imgSize_rgb, &(levels[0]), static_cast<int>(levels.size()), lowLevelEngine);
#endif
			break;
	}

	if (ret == nullptr) DIEWITHEXCEPTION("Failed to make colour tracker");
	return ret;
}

/**
 * \brief Makes an ICP tracker.
 */
ITMTracker*
ITMTrackerFactory::MakeICPTracker(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, ITMLibSettings::DeviceType deviceType,
               const ORUtils::KeyValueConfig& cfg,
               const ITMLowLevelEngine* lowLevelEngine, ITMIMUCalibrator* imuCalibrator,
               const ITMSceneParams* sceneParams)
{
	ITMICPTracker::Parameters parameters;

	int verbose = 0;
	if (cfg.getProperty("help") != nullptr) if (verbose < 10) verbose = 10;

	const char* levelSetup = "rrrbb";
	cfg.parseStrProperty("levels", "resolution hierarchy levels", levelSetup, verbose);
	parameters.levels = parseLevelConfig(levelSetup);

	cfg.parseBoolProperty("useDepth", "use depth based tracking", parameters.useDepth, verbose);
	cfg.parseBoolProperty("useColour", "use colour based tracking", parameters.useColour, verbose);
	cfg.parseFltProperty("colourWeight",
	                     "weight used to scale colour errors and jacobians when both useColour and useWeights are set",
	                     parameters.colourWeight, verbose);
	const char* colourMode = "f2f";
	cfg.parseStrProperty("colourMode",
	                     "weight used to scale colour errors and jacobians when both useColour and useWeights are set",
	                     colourMode, verbose);
	parameters.colourMode = parameters.ColourModeFromString(colourMode);
	cfg.parseBoolProperty("optimizeScale", "optimize in sim3 to also estimate scale factor", parameters.optimizeScale, verbose);
	cfg.parseFltProperty("minstep", "step size threshold for convergence", parameters.smallStepSizeCriterion, verbose);
	cfg.parseFltProperty("outlierDistanceC", "distance outlier threshold at coarsest level",
	                     parameters.outlierDistanceCoarse, verbose);
	cfg.parseFltProperty("outlierDistanceF", "distance outlier threshold at finest level",
	                     parameters.outlierDistanceFine, verbose);
	cfg.parseFltProperty("outlierColourC", "colour outlier threshold at coarsest level", parameters.outlierColourCoarse,
	                     verbose);
	cfg.parseFltProperty("outlierColourF", "colour outlier threshold at finest level", parameters.outlierColourFine,
	                     verbose);
	cfg.parseFltProperty("minColourGradient", "minimum colour gradient for a pixel to be used in the tracking",
	                     parameters.minColourGradient, verbose);
	cfg.parseIntProperty("numiterC", "maximum number of iterations at coarsest level", parameters.numIterationsCoarse,
	                     verbose);
	cfg.parseIntProperty("numiterF", "maximum number of iterations at finest level", parameters.numIterationsFine,
	                     verbose);
	cfg.parseFltProperty("failureDec", "threshold for the failure detection", parameters.failureDetectorThreshold,
	                     verbose);

	ITMICPTracker* ret = nullptr;
	switch (deviceType)
	{
		case ITMLibSettings::DEVICE_CPU:
			ret = new ITMICPTracker_CPU(imgSize_d, imgSize_rgb, parameters, lowLevelEngine);
			break;
		case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
			ret = new ITMICPTracker_CUDA(imgSize_d, imgSize_rgb, parameters, lowLevelEngine);
#endif
			break;
	}

	if (ret == nullptr) DIEWITHEXCEPTION("Failed to make ICP tracker");
	return ret;
}

/**
* \brief Makes an Extended tracker.
*/
ITMTracker*
ITMTrackerFactory::MakeExtendedTracker(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, ITMLibSettings::DeviceType deviceType,
                    const ORUtils::KeyValueConfig& cfg,
                    const ITMLowLevelEngine* lowLevelEngine, ITMIMUCalibrator* imuCalibrator,
                    const ITMSceneParams* sceneParams)
{
	const char* levelSetup = "rrbb";
	bool useDepth = true;
	bool useColour = false;
	float colourWeight = 0.3f;
	float smallStepSizeCriterion = 1e-4f;
	float outlierSpaceDistanceFine = 0.004f;
	float outlierSpaceDistanceCoarse = 0.1f;
	float outlierColourDistanceFine = 0.175f;
	float outlierColourDistanceCoarse = 0.005f;
	float failureDetectorThd = 3.0f;
	float minColourGradient = 0.01f;
	float tukeyCutOff = 8.0f;
	int framesToSkip = 20;
	int framesToWeight = 50;
	int numIterationsCoarse = 20;
	int numIterationsFine = 20;

	int verbose = 0;
	if (cfg.getProperty("help") != nullptr) if (verbose < 10) verbose = 10;
	cfg.parseStrProperty("levels", "resolution hierarchy levels", levelSetup, verbose);
	std::vector<TrackerIterationType> levels = parseLevelConfig(levelSetup);

	cfg.parseBoolProperty("useDepth", "use ICP based tracking", useDepth, verbose);
	cfg.parseBoolProperty("useColour", "use colour based tracking", useColour, verbose);
	cfg.parseFltProperty("colourWeight",
	                     "weight used to scale colour errors and jacobians when both useColour and useWeights are set",
	                     colourWeight, verbose);
	cfg.parseFltProperty("minstep", "step size threshold for convergence", smallStepSizeCriterion, verbose);
	cfg.parseFltProperty("outlierSpaceC", "space outlier threshold at coarsest level", outlierSpaceDistanceCoarse,
	                     verbose);
	cfg.parseFltProperty("outlierSpaceF", "space outlier threshold at finest level", outlierSpaceDistanceFine, verbose);
	cfg.parseFltProperty("outlierColourC", "colour outlier threshold at coarsest level", outlierColourDistanceCoarse,
	                     verbose);
	cfg.parseFltProperty("outlierColourF", "colour outlier threshold at finest level", outlierColourDistanceFine,
	                     verbose);
	cfg.parseFltProperty("minColourGradient", "minimum colour gradient for a pixel to be used in the tracking",
	                     minColourGradient, verbose);
	cfg.parseIntProperty("numiterC", "maximum number of iterations at coarsest level", numIterationsCoarse, verbose);
	cfg.parseIntProperty("numiterF", "maximum number of iterations at finest level", numIterationsFine, verbose);
	cfg.parseFltProperty("tukeyCutOff", "cutoff for the tukey m-estimator", tukeyCutOff, verbose);
	cfg.parseIntProperty("framesToSkip", "number of frames to skip before depth pixel is used for tracking",
	                     framesToSkip, verbose);
	cfg.parseIntProperty("framesToWeight", "number of frames to weight each depth pixel for before using it fully",
	                     framesToWeight, verbose);
	cfg.parseFltProperty("failureDec", "threshold for the failure detection", failureDetectorThd, verbose);

	ITMExtendedTracker* ret = nullptr;
	switch (deviceType)
	{
		case ITMLibSettings::DEVICE_CPU:
			ret = new ITMExtendedTracker_CPU(imgSize_d,
			                                 imgSize_rgb,
			                                 useDepth,
			                                 useColour,
			                                 colourWeight,
			                                 &(levels[0]),
			                                 static_cast<int>(levels.size()),
			                                 smallStepSizeCriterion,
			                                 failureDetectorThd,
			                                 sceneParams->viewFrustum_min,
			                                 sceneParams->viewFrustum_max,
			                                 minColourGradient,
			                                 tukeyCutOff,
			                                 framesToSkip,
			                                 framesToWeight,
			                                 lowLevelEngine);
			break;
		case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
			ret = new ITMExtendedTracker_CUDA(imgSize_d,
			                                  imgSize_rgb,
			                                  useDepth,
			                                  useColour,
			                                  colourWeight,
			                                  &(levels[0]),
			                                  static_cast<int>(levels.size()),
			                                  smallStepSizeCriterion,
			                                  failureDetectorThd,
			                                  sceneParams->viewFrustum_min,
			                                  sceneParams->viewFrustum_max,
			                                  minColourGradient,
			                                  tukeyCutOff,
			                                  framesToSkip,
			                                  framesToWeight,
			                                  lowLevelEngine);
#endif
			break;
	}

	if (ret == nullptr) DIEWITHEXCEPTION("Failed to make extended tracker");
	ret->SetupLevels(numIterationsCoarse, numIterationsFine, outlierSpaceDistanceCoarse, outlierSpaceDistanceFine,
	                 outlierColourDistanceCoarse, outlierColourDistanceFine);
	return ret;
}

/**
 * \brief Makes an IMU tracker.
 */
ITMTracker*
ITMTrackerFactory::MakeIMUTracker(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, ITMLibSettings::DeviceType deviceType,
               const ORUtils::KeyValueConfig& cfg,
               const ITMLowLevelEngine* lowLevelEngine, ITMIMUCalibrator* imuCalibrator,
               const ITMSceneParams* sceneParams)
{
	const char* levelSetup = "tb";
	float smallStepSizeCriterion = 1e-3f;
	float outlierDistanceFine = 0.005f;
	float outlierDistanceCoarse = 0.01f;
	float failureDetectorThd = 3.0f;
	int numIterationsCoarse = 4;
	int numIterationsFine = 2;

	int verbose = 0;
	if (cfg.getProperty("help") != nullptr) if (verbose < 10) verbose = 10;
	cfg.parseStrProperty("levels", "resolution hierarchy levels", levelSetup, verbose);
	std::vector<TrackerIterationType> levels = parseLevelConfig(levelSetup);

	cfg.parseFltProperty("minstep", "step size threshold for convergence", smallStepSizeCriterion, verbose);
	cfg.parseFltProperty("outlierC", "outlier threshold at coarsest level", outlierDistanceCoarse, verbose);
	cfg.parseFltProperty("outlierF", "outlier threshold at finest level", outlierDistanceFine, verbose);
	cfg.parseIntProperty("numiterC", "maximum number of iterations at coarsest level", numIterationsCoarse, verbose);
	cfg.parseIntProperty("numiterF", "maximum number of iterations at finest level", numIterationsFine, verbose);
	cfg.parseFltProperty("failureDec", "threshold for the failure detection", failureDetectorThd, verbose);

	ITMICPTracker* dTracker = nullptr;
	switch (deviceType)
	{
		case ITMLibSettings::DEVICE_CPU:
//			dTracker = new ITMICPTracker_CPU(imgSize_d, &(levels[0]), static_cast<int>(levels.size()), smallStepSizeCriterion, failureDetectorThd, lowLevelEngine);
			break;
		case ITMLibSettings::DEVICE_CUDA:
#ifndef COMPILE_WITHOUT_CUDA
//			dTracker = new ITMICPTracker_CUDA(imgSize_d, &(levels[0]), static_cast<int>(levels.size()), smallStepSizeCriterion, failureDetectorThd, lowLevelEngine);
#endif
			break;
		default:
			break;
	}

	if (dTracker == nullptr) DIEWITHEXCEPTION("Failed to make IMU tracker");

	ITMCompositeTracker* compositeTracker = new ITMCompositeTracker;
	compositeTracker->AddTracker(new ITMIMUTracker(imuCalibrator));
	compositeTracker->AddTracker(dTracker);
	return compositeTracker;
}

/**
* \brief Makes an Extended IMU tracker.
*/
ITMTracker*
ITMTrackerFactory::MakeExtendedIMUTracker(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, ITMLibSettings::DeviceType deviceType,
                       const ORUtils::KeyValueConfig& cfg,
                       const ITMLowLevelEngine* lowLevelEngine, ITMIMUCalibrator* imuCalibrator,
                       const ITMSceneParams* sceneParams)
{
	ITMTracker* dTracker = MakeExtendedTracker(imgSize_rgb, imgSize_d, deviceType, cfg,
	                                           lowLevelEngine, imuCalibrator, sceneParams);
	if (dTracker == nullptr) DIEWITHEXCEPTION("Failed to make extended tracker"); // Should never happen though

	ITMCompositeTracker* compositeTracker = new ITMCompositeTracker;
	compositeTracker->AddTracker(new ITMIMUTracker(imuCalibrator));
	compositeTracker->AddTracker(dTracker);
	return compositeTracker;
}

/**
 * \brief Makes a file based tracker.
 */
ITMTracker*
ITMTrackerFactory::MakeFileBasedTracker(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, ITMLibSettings::DeviceType deviceType,
                     const ORUtils::KeyValueConfig& cfg,
                     const ITMLowLevelEngine* lowLevelEngine, ITMIMUCalibrator* imuCalibrator,
                     const ITMSceneParams* sceneParams)
{
	int verbose = 0;
	if (cfg.getProperty("help") && verbose < 10) verbose = 10;

	const char* fileMask = "";
	int initialFrameNo = 0;
	cfg.parseStrProperty("mask", "mask for the saved pose text files", fileMask, verbose);
	cfg.parseIntProperty("initialFrameNo", "initial frame index to use for tracking", initialFrameNo, verbose);

	return new ITMFileBasedTracker(fileMask, initialFrameNo);
}

/**
 * \brief Makes a force fail tracker.
 */
ITMTracker*
ITMTrackerFactory::MakeForceFailTracker(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, ITMLibSettings::DeviceType deviceType,
                     const ORUtils::KeyValueConfig& cfg,
                     const ITMLowLevelEngine* lowLevelEngine, ITMIMUCalibrator* imuCalibrator,
                     const ITMSceneParams* sceneParams)
{
	return new ITMForceFailTracker;
}

} // namespace ITMLib
