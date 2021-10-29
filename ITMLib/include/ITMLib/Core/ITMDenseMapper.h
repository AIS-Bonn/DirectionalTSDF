// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <memory>
#include <ITMLib/Engines/ITMSceneReconstructionEngine.h>
#include <ITMLib/Utils/ITMLibSettings.h>

namespace ITMLib
{
/** \brief
*/
class ITMDenseMapper
{
private:
	IITMSceneReconstructionEngine* sceneRecoEngine;

	ITMLibSettings::SwappingMode swappingMode;

public:
	void ResetScene(Scene* scene) const;

	/// Process a single frame
	void ProcessFrame(const ITMView* view, const ITMTrackingState* trackingState, Scene* scene,
	                  ITMRenderState* renderState_live);

	/// Update the visible list (this can be called to update the visible list when fusion is turned off)
	void UpdateVisibleList(const ITMView* view, const ITMTrackingState* trackingState, Scene* scene,
	                       ITMRenderState* renderState);

	IITMSceneReconstructionEngine* GetSceneReconstructionEngine()
	{
		return sceneRecoEngine;
	}

	/** \brief Constructor
			Ommitting a separate image size for the depth images
			will assume same resolution as for the RGB images.
	*/
	explicit ITMDenseMapper(const std::shared_ptr<const ITMLibSettings>& settings);

	~ITMDenseMapper();
};
}
