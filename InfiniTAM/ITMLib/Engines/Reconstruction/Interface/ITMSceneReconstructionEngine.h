// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <math.h>

#include "ITMLib/Objects/RenderStates/ITMRenderState.h"
#include "ITMLib/Objects/Scene/ITMScene.h"
#include "ITMLib/Objects/Tracking/ITMTrackingState.h"
#include "ITMLib/Objects/Views/ITMView.h"
#include "ITMLib/Utils/ITMLibSettings.h"

namespace ITMLib
{
	/** \brief
	    Interface to engines implementing the main KinectFusion
	    depth integration process.

	    These classes basically manage
	    an ITMLib::Objects::ITMScene and fuse new image information
	    into them.
	*/
	template<class TVoxel, class TIndex>
	class ITMSceneReconstructionEngine
	{
	public:
		ITMSceneReconstructionEngine(ITMLibSettings::TSDFMode tsdfMode,
			ITMLibSettings::FusionMode fusionMode, ITMLibSettings::FusionMetric fusionMetric)
			: tsdfMode(tsdfMode), fusionMode(fusionMode), fusionMetric(fusionMetric)
		{ }

		/** Clear and reset a scene to set up a new empty
		    one.
		*/
		virtual void ResetScene(ITMScene<TVoxel, TIndex> *scene) = 0;

		/** Given a view with a new depth image, compute the
		    visible blocks, allocate them and update the hash
		    table so that the new image data can be integrated.
		*/
		virtual void AllocateSceneFromDepth(ITMScene<TVoxel,TIndex> *scene, const ITMView *view, const ITMTrackingState *trackingState,
			const ITMRenderState *renderState, bool onlyUpdateVisibleList = false, bool resetVisibleList = false) = 0;

		/** Update the voxel blocks by integrating depth and
		    possibly colour information from the given view.
		*/
		virtual void IntegrateIntoScene(ITMScene<TVoxel,TIndex> *scene, const ITMView *view, const ITMTrackingState *trackingState,
			const ITMRenderState *renderState) = 0;

		ITMSceneReconstructionEngine(void) { }
		virtual ~ITMSceneReconstructionEngine(void) { }

	protected:
		ITMLibSettings::TSDFMode tsdfMode;
		ITMLibSettings::FusionMode fusionMode;
		ITMLibSettings::FusionMetric fusionMetric;
	};

	enum HashEntryAllocType : uchar
	{
		ALLOCATED = 0,
		ALLOCATE_ORDERED = 1, // entry requires allocation in ordered list
		ALLOCATE_EXCESS = 2  // entry requires allocation in excess list
	};

	enum HashEntryVisibilityType : uchar
	{
		INVISIBLE = 0,
		VISIBLE_IN_MEMORY = 1,
		VISIBLE_STREAMED_OUT = 2,
		PREVIOUSLY_VISIBLE = 3 // visible at previous frame and unstreamed
	};
}
