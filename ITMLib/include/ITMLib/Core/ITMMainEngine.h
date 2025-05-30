// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <ITMLib/Engines/ITMVisualisationEngine.h>
#include <ITMLib/Objects/Stats/ITMRenderError.h>
#include <ITMLib/Objects/Misc/ITMIMUMeasurement.h>
#include <ITMLib/Objects/Stats/ITMTimeStats.h>
#include <ITMLib/Trackers/ITMTracker.h>
#include <ITMLib/Utils/ITMLibSettings.h>

#include <utility>

/** \mainpage
    This is the API reference documentation for InfiniTAM. For a general
    overview additional documentation can be found in the included Technical
    Report.

    For use of ITMLib in your own project, the class
    @ref ITMLib::Engine::ITMMainEngine should be the main interface and entry
    point to the library.
*/

namespace ITMLib
{
/** \brief
		Main engine, that instantiates all the other engines and
		provides a simplified interface to them.

		This class is the main entry point to the ITMLib library
		and basically performs the whole KinectFusion algorithm.
		It stores the latest image internally, as well as the 3D
		world model and additionally it keeps track of the camera
		pose.

		The intended use is as follows:
		-# Create an ITMMainEngine specifying the internal settings,
			 camera parameters and image sizes
		-# Get the pointer to the internally stored images with
			 @ref GetView() and write new image information to that
			 memory
		-# Call the method @ref ProcessFrame() to track the camera
			 and integrate the new information into the world model
		-# Optionally access the rendered reconstruction or another
			 image for visualisation using @ref GetImage()
		-# Iterate the above three steps for each image in the
			 sequence

		To access the internal information, look at the member
		variables @ref trackingState and @ref scene.
*/
class ITMMainEngine
{
public:
	enum GetImageType
	{
		InfiniTAM_IMAGE_ORIGINAL_RGB,
		InfiniTAM_IMAGE_ORIGINAL_DEPTH,
		InfiniTAM_IMAGE_SCENERAYCAST,
		InfiniTAM_IMAGE_COLOUR_FROM_VOLUME,
		InfiniTAM_IMAGE_COLOUR_FROM_NORMAL,
		InfiniTAM_IMAGE_COLOUR_FROM_CONFIDENCE,
		InfiniTAM_IMAGE_COLOUR_FROM_DEPTH,
		InfiniTAM_IMAGE_COLOUR_FROM_ICP_ERROR,
		InfiniTAM_IMAGE_FREECAMERA_SHADED,
		InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME,
		InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL,
		InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_CONFIDENCE,
		InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_DEPTH,
		InfiniTAM_IMAGE_UNKNOWN
	};

	/// Gives access to the current input frame
	virtual ITMView* GetView() = 0;

	/// Gives access to the current camera pose and additional tracking information
	virtual ITMTrackingState* GetTrackingState() = 0;

	virtual ITMRenderState* GetRenderState() = 0;

	virtual ITMRenderState* GetRenderStateFreeview() = 0;

	virtual ITMRenderError ComputeICPError()
	{ return ITMRenderError(); };

	virtual ITMRenderError ComputePhotometricError()
	{ return ITMRenderError(); };

	/// Process a frame with rgb and depth images and optionally a corresponding imu measurement
	virtual ITMTrackingState::TrackingResult
	ProcessFrame(ITMUChar4Image* rgbImage, ITMShortImage* rawDepthImage, ITMIMUMeasurement* imuMeasurement,
	             const ORUtils::SE3Pose* pose) = 0;

	/// Get a result image as output
	[[nodiscard]] virtual Vector2i GetImageSize() const = 0;

	virtual void
	GetImage(ITMUChar4Image* out, const GetImageType getImageType, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics,
	         bool normalsFromSDF) = 0;

	virtual void
	GetPointCloud(ITMPointCloud* out, const ORUtils::SE3Pose* pose, const ITMIntrinsics* intrinsics, bool normalsFromSDF) = 0;

	/// Extracts a mesh from the current scene and saves it to the model file specified by the file name
	virtual void SaveSceneToMesh(const char* fileName)
	{};

	/// save and load the full scene and relocaliser (if any) to/from file
	virtual void SaveToFile()
	{};

	virtual void LoadFromFile()
	{};

	virtual ~ITMMainEngine() = default;

	explicit ITMMainEngine(std::shared_ptr<const ITMLibSettings> settings) : settings(std::move(settings)) { }

	virtual const unsigned int* GetAllocationsPerDirection() = 0;

	[[nodiscard]] const ITMTimeStats& GetTimeStats() const
	{
		return timeStats;
	}

	[[nodiscard]] inline std::shared_ptr<const ITMLibSettings> GetSettings() const
    {
	    return settings;
    }


	/**
	 * Converts GetImageType to suitable RenderImageType to use with visualization engine.
	 * @param getImageType
	 * @param normalsFromSDF whether to use SDF or image points to interpolate normals
	 * @return
	 */
	static IITMVisualisationEngine::RenderImageType ImageTypeToRenderType(GetImageType getImageType, bool normalsFromSDF)
	{
		if (normalsFromSDF)
		{
			switch (getImageType)
			{
				case ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_VOLUME:
				case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME:
					return IITMVisualisationEngine::RENDER_COLOUR;
				case ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_NORMAL:
				case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL:
					return IITMVisualisationEngine::RENDER_NORMAL_SDFNORMAL;
				case ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_CONFIDENCE:
				case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_CONFIDENCE:
					return IITMVisualisationEngine::RENDER_CONFIDENCE_SDFNORMAL;
				case ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_DEPTH:
				case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_DEPTH:
					return IITMVisualisationEngine::RENDER_DEPTH_COLOUR;
				default:
					return IITMVisualisationEngine::RENDER_DEPTH_SDFNORMAL;
			}
		}
		switch (getImageType)
		{
			case ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_VOLUME:
			case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME:
				return IITMVisualisationEngine::RENDER_COLOUR;
			case ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_NORMAL:
			case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL:
				return IITMVisualisationEngine::RENDER_NORMAL_IMAGENORMAL;
			case ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_CONFIDENCE:
			case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_CONFIDENCE:
				return IITMVisualisationEngine::RENDER_CONFIDENCE_IMAGENORMAL;
			case ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_DEPTH:
			case ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_DEPTH:
				return IITMVisualisationEngine::RENDER_DEPTH_COLOUR;
			default:
				return IITMVisualisationEngine::RENDER_DEPTH_IMAGENORMAL;
		}
	}

protected:
    std::shared_ptr<const ITMLibSettings> settings;
    ITMTimeStats timeStats;
};

} // namespace ITMLib
