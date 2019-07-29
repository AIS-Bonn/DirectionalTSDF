// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <memory>
#include <math.h>

#include "ITMLib/Objects/RenderStates/ITMRenderState.h"
#include "ITMLib/Objects/Scene/ITMScene.h"
#include "ITMLib/Objects/Tracking/ITMTrackingState.h"
#include "ITMLib/Objects/Views/ITMView.h"
#include "ITMLib/Utils/ITMLibSettings.h"
#include "ITMLib/Objects/Stats/ITMReconstructionTimeStats.h"

namespace ITMLib
{
	typedef struct
	{
		float sdfSum;
		float weightSum;

		_CPU_AND_GPU_CODE_
		inline void reset()
		{
			sdfSum = 0.0f;
			weightSum = 0.0f;
		}

		_CPU_AND_GPU_CODE_
		inline void update(float sdf, float weight)
		{
#ifdef __CUDA_ARCH__
			atomicAdd(&sdfSum, weight * sdf);
			atomicAdd(&weightSum, weight);
#else
			sdfSum += weight * sdf;
			weightSum += weight;
#endif
		}
	} VoxelRayCastingSum;

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
		explicit ITMSceneReconstructionEngine(std::shared_ptr<const ITMLibSettings> settings)
			:settings(std::move(settings)), entriesRayCasting(nullptr)
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
		void IntegrateIntoScene(ITMScene<TVoxel, TIndex>* scene, const ITMView* view,
			const ITMTrackingState* trackingState, const ITMRenderState* renderState)
		{
			if (this->settings->fusionParams.fusionMode == FusionMode::FUSIONMODE_RAY_CASTING)
			{
				IntegrateIntoSceneRayCasting(scene, view, trackingState, renderState);
			}
			else
			{
				IntegrateIntoSceneVoxelProjection(scene, view, trackingState, renderState);
			}
		}

		ITMReconstructionTimeStats &GetTimeStats()
		{
			return timeStats;
		}

		const ITMReconstructionTimeStats &GetTimeStats() const
		{
			return timeStats;
		}

		ITMSceneReconstructionEngine(void) { }
		virtual ~ITMSceneReconstructionEngine(void) { }

	protected:
		std::shared_ptr<const ITMLibSettings> settings;

		/**
		 * Per-hash entry summation values for ray casting fusion update
		 */
		ORUtils::MemoryBlock<VoxelRayCastingSum> *entriesRayCasting;

		ITMReconstructionTimeStats timeStats;

		virtual void IntegrateIntoSceneVoxelProjection(ITMScene<TVoxel, TIndex>* scene, const ITMView* view,
		                                               const ITMTrackingState* trackingState,
		                                               const ITMRenderState* renderState) = 0;

		virtual void IntegrateIntoSceneRayCasting(ITMScene<TVoxel,TIndex> *scene, const ITMView *view,
		                                          const ITMTrackingState *trackingState, const ITMRenderState *renderState) = 0;
	};
}
