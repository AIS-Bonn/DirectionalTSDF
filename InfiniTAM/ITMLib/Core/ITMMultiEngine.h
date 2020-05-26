// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "ITMMainEngine.h"
#include "ITMTrackingController.h"
#include "../Engines/LowLevel/Interface/ITMLowLevelEngine.h"
#include "../Engines/ViewBuilding/Interface/ITMViewBuilder.h"
#include "../Objects/Misc/ITMIMUCalibrator.h"
#include "../../FernRelocLib/Relocaliser.h"

#include "../Engines/MultiScene/ITMActiveMapManager.h"
#include "../Engines/MultiScene/ITMGlobalAdjustmentEngine.h"
#include "../Engines/Visualisation/Interface/ITMMultiVisualisationEngine.h"
#include "../Engines/Meshing/ITMMultiMeshingEngineFactory.h"

#include <vector>

namespace ITMLib
{
	/** \brief
	*/
	template <typename TVoxel, typename TIndex>
	class ITMMultiEngine : public ITMMainEngine
	{
	private:
		std::shared_ptr<const ITMLibSettings> settings;

		ITMLowLevelEngine *lowLevelEngine;
		ITMVisualisationEngine<TVoxel, TIndex> *visualisationEngine;
		ITMMultiVisualisationEngine<TVoxel, TIndex> *multiVisualisationEngine;

		ITMMultiMeshingEngine<TVoxel, TIndex> *meshingEngine;

		ITMViewBuilder *viewBuilder;
		ITMTrackingController *trackingController;
		ITMTracker *tracker;
		ITMIMUCalibrator *imuCalibrator;
		ITMDenseMapper<TVoxel, TIndex> *denseMapper;

		FernRelocLib::Relocaliser<float> *relocaliser;

		ITMVoxelMapGraphManager<TVoxel, TIndex> *mapManager;
		ITMActiveMapManager *mActiveDataManager;
		ITMGlobalAdjustmentEngine *mGlobalAdjustmentEngine;
		bool mScheduleGlobalAdjustment;

		Vector2i trackedImageSize;
		ITMRenderState *renderState_freeview;
		ITMRenderState *renderState_multiscene;
		int freeviewLocalMapIdx;

		/// Pointer for storing the current input frame
		ITMView *view;
	public:
		ITMView* GetView() override { return view; }

		ITMTrackingState* GetTrackingState() override;

		const unsigned int* GetAllocationsPerDirection() override;

		/// Process a frame with rgb and depth images and (optionally) a corresponding imu measurement
		ITMTrackingState::TrackingResult ProcessFrame(ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage, ITMIMUMeasurement *imuMeasurement = nullptr, const ORUtils::SE3Pose* pose = nullptr) override;

		/// Get a result image as output
		Vector2i GetImageSize() const override;

		void GetImage(ITMUChar4Image *out, GetImageType getImageType, ORUtils::SE3Pose *pose = nullptr, ITMIntrinsics *intrinsics = nullptr) override;

		void changeFreeviewLocalMapIdx(ORUtils::SE3Pose *pose, int newIdx);
		void setFreeviewLocalMapIdx(int newIdx)
		{
			freeviewLocalMapIdx = newIdx;
		}
		[[nodiscard]]
		int getFreeviewLocalMapIdx() const
		{
			return freeviewLocalMapIdx;
		}
		[[nodiscard]]
		int findPrimaryLocalMapIdx() const
		{
			return mActiveDataManager->findPrimaryLocalMapIdx();
		}

		/// Extracts a mesh from the current scene and saves it to the model file specified by the file name
		void SaveSceneToMesh(const char *fileName) override;

		/// save and load the full scene and relocaliser (if any) to/from file
		void SaveToFile() override;
		void LoadFromFile() override;

		//void writeFullTrajectory(void) const;
		//void SaveSceneToMesh(const char *objFileName);

		/** \brief Constructor
			Ommitting a separate image size for the depth images
			will assume same resolution as for the RGB images.
		*/
		ITMMultiEngine(const std::shared_ptr<const ITMLibSettings>& settings, const ITMRGBDCalib &calib, Vector2i imgSize_rgb, Vector2i imgSize_d = Vector2i(-1, -1));
		~ITMMultiEngine(void);
	};
}
