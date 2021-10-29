// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <stdexcept>
#include <memory>
#include <vector>

#include <ITMLib/Trackers/ITMCompositeTracker.h>
#include <ITMLib/Trackers/ITMIMUTracker.h>
#include <ITMLib/Trackers/ITMFileBasedTracker.h>
#include <ITMLib/Trackers/ITMForceFailTracker.h>
#include <ITMLib/Trackers/ITMTracker.h>
#include <ITMLib/Engines/ITMLowLevelEngine.h>
#include <ITMLib/Utils/ITMLibSettings.h>

#include <ORUtils/KeyValueConfig.h>
#include <ITMLib/Objects/Tracking/TrackerIterationType.h>

namespace ITMLib
{
/**
 * \brief An instance of this class can be used to construct trackers.
 */
class ITMTrackerFactory
{
private:
	//#################### TYPEDEFS ####################
	typedef ITMTracker* MakerFunc(const Vector2i&, const Vector2i&, ITMLibSettings::DeviceType,
	                              const ORUtils::KeyValueConfig&, const ITMLowLevelEngine*, ITMIMUCalibrator*,
	                              const ITMSceneParams*);

	/// Tracker types
	typedef enum
	{
		//! Identifies a tracker based on colour image
		TRACKER_COLOR,
		//! Identifies a tracker based on depth image
		TRACKER_ICP,
		//! Identifies a tracker based on depth and color image with various extensions
		TRACKER_EXTENDED,
		//! Identifies a tracker reading poses from text files
		TRACKER_FILE,
		//! Identifies a tracker based on depth image and IMU measurement
		TRACKER_IMU,
		//! Identifies a tracker based on depth and colour images and IMU measurement
		TRACKER_EXTENDEDIMU,
		//! Identifies a tracker that forces tracking to fail
		TRACKER_FORCEFAIL,
	} TrackerType;

	struct Maker
	{
		const char* id;
		const char* description;
		TrackerType type;
		MakerFunc* make;

		Maker(const char* _id, const char* _desc, TrackerType _type, MakerFunc* _make)
			: id(_id), description(_desc), type(_type), make(_make)
		{}
	};

	//#################### PRIVATE VARIABLES ####################
	/** A list of maker functions for the various tracker types. */
	std::vector<Maker> makers;

	//################## SINGLETON IMPLEMENTATION ##################
	/**
	 * \brief Constructs a tracker factory.
	 */
	ITMTrackerFactory()
	{
		makers.emplace_back("rgb", "Colour based tracker", TRACKER_COLOR, &MakeColourTracker);
		makers.emplace_back("icp", "Depth based ICP tracker", TRACKER_ICP, &MakeICPTracker);
		makers.emplace_back("extended", "Depth + colour based tracker", TRACKER_EXTENDED, &MakeExtendedTracker);
		makers.emplace_back("file", "File based tracker", TRACKER_FILE, &MakeFileBasedTracker);
		makers.emplace_back("imuicp", "Combined IMU and depth based ICP tracker", TRACKER_IMU, &MakeIMUTracker);
		makers.emplace_back("extendedimu", "Combined IMU and depth + colour ICP tracker", TRACKER_EXTENDEDIMU,
		                    &MakeExtendedIMUTracker);
		makers.emplace_back("forcefail", "Force fail tracker", TRACKER_FORCEFAIL, &MakeForceFailTracker);
	}

public:
	/**
	 * \brief Gets the singleton instance for the current set of template parameters.
	 */
	static ITMTrackerFactory& Instance()
	{
		static ITMTrackerFactory s_instance;
		return s_instance;
	}

	//################## PUBLIC MEMBER FUNCTIONS ##################
public:
	/**
	 * \brief Makes a tracker of the type specified in the trackerConfig string.
	 */
	ITMTracker* Make(ITMLibSettings::DeviceType deviceType, const char* trackerConfig, const Vector2i& imgSize_rgb,
	                 const Vector2i& imgSize_d, const ITMLowLevelEngine* lowLevelEngine,
	                 ITMIMUCalibrator* imuCalibrator, const ITMSceneParams* sceneParams) const;

	/**
	 * \brief Makes a tracker of the type specified in the settings.
	 */
	ITMTracker*
	Make(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, const std::shared_ptr<const ITMLibSettings>& settings,
	     const ITMLowLevelEngine* lowLevelEngine,
	     ITMIMUCalibrator* imuCalibrator, const ITMSceneParams* sceneParams) const;

	//#################### PUBLIC STATIC MEMBER FUNCTIONS ####################
	static std::vector<TrackerIterationType> parseLevelConfig(const char* str);

	/**
	 * \brief Makes a colour tracker.
	 */
	static ITMTracker*
	MakeColourTracker(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, ITMLibSettings::DeviceType deviceType,
	                  const ORUtils::KeyValueConfig& cfg,
	                  const ITMLowLevelEngine* lowLevelEngine, ITMIMUCalibrator* imuCalibrator,
	                  const ITMSceneParams* sceneParams);

	/**
	 * \brief Makes an ICP tracker.
	 */
	static ITMTracker*
	MakeICPTracker(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, ITMLibSettings::DeviceType deviceType,
	               const ORUtils::KeyValueConfig& cfg,
	               const ITMLowLevelEngine* lowLevelEngine, ITMIMUCalibrator* imuCalibrator,
	               const ITMSceneParams* sceneParams);

	/**
	* \brief Makes an Extended tracker.
	*/
	static ITMTracker*
	MakeExtendedTracker(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, ITMLibSettings::DeviceType deviceType,
	                    const ORUtils::KeyValueConfig& cfg,
	                    const ITMLowLevelEngine* lowLevelEngine, ITMIMUCalibrator* imuCalibrator,
	                    const ITMSceneParams* sceneParams);
	/**
	 * \brief Makes an IMU tracker.
	 */
	static ITMTracker*
	MakeIMUTracker(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, ITMLibSettings::DeviceType deviceType,
	               const ORUtils::KeyValueConfig& cfg,
	               const ITMLowLevelEngine* lowLevelEngine, ITMIMUCalibrator* imuCalibrator,
	               const ITMSceneParams* sceneParams);

	/**
	* \brief Makes an Extended IMU tracker.
	*/
	static ITMTracker*
	MakeExtendedIMUTracker(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, ITMLibSettings::DeviceType deviceType,
	                       const ORUtils::KeyValueConfig& cfg,
	                       const ITMLowLevelEngine* lowLevelEngine, ITMIMUCalibrator* imuCalibrator,
	                       const ITMSceneParams* sceneParams);

	/**
	 * \brief Makes a file based tracker.
	 */
	static ITMTracker*
	MakeFileBasedTracker(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, ITMLibSettings::DeviceType deviceType,
	                     const ORUtils::KeyValueConfig& cfg,
	                     const ITMLowLevelEngine* lowLevelEngine, ITMIMUCalibrator* imuCalibrator,
	                     const ITMSceneParams* sceneParams);

	/**
	 * \brief Makes a force fail tracker.
	 */
	static ITMTracker*
	MakeForceFailTracker(const Vector2i& imgSize_rgb, const Vector2i& imgSize_d, ITMLibSettings::DeviceType deviceType,
	                     const ORUtils::KeyValueConfig& cfg,
	                     const ITMLowLevelEngine* lowLevelEngine, ITMIMUCalibrator* imuCalibrator,
	                     const ITMSceneParams* sceneParams);
};
}
