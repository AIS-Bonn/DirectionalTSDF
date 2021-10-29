// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <ITMLib/Trackers/ITMTracker.h>
#include <ITMLib/Engines/ITMLowLevelEngine.h>
#include <ITMLib/Objects/Tracking/ITMImageHierarchy.h>
#include <ITMLib/Objects/Tracking/ITMViewHierarchyLevel.h>
#include <ITMLib/Objects/Tracking/TrackerIterationType.h>

namespace ITMLib
{
/** Base class for engines performing point based colour
		tracking. Implementations would typically project down a
		point cloud into observed images and try to minimize the
		reprojection error.
*/
class ITMColorTracker : public ITMTracker
{
private:
	const ITMLowLevelEngine* lowLevelEngine;

	void PrepareForEvaluation(const ITMView* view);

protected:
	TrackerIterationType iterationType;
	ITMTrackingState* trackingState;
	const ITMView* view;
	ITMImageHierarchy<ITMViewHierarchyLevel>* viewHierarchy;
	int levelId;

	int countedPoints_valid;
public:
	class EvaluationPoint
	{
	public:
		[[nodiscard]] float f() const
		{ return cacheF; }

		const float* nabla_f()
		{
			if (cacheNabla == nullptr) computeGradients(false);
			return cacheNabla;
		}

		const float* hessian_GN()
		{
			if (cacheHessian == nullptr) computeGradients(true);
			return cacheHessian;
		}

		[[nodiscard]] const ORUtils::SE3Pose& getParameter() const
		{ return *mPara; }

		EvaluationPoint(ORUtils::SE3Pose* pos, const ITMColorTracker* f_parent);

		~EvaluationPoint()
		{
			delete mPara;
			delete[] cacheNabla;
			delete[] cacheHessian;
		}

		[[nodiscard]] int getNumValidPoints() const
		{ return mValidPoints; }

	protected:
		void computeGradients(bool requiresHessian);

		ORUtils::SE3Pose* mPara;
		const ITMColorTracker* mParent;

		float cacheF;
		float* cacheNabla;
		float* cacheHessian;
		int mValidPoints;
	};

	EvaluationPoint* evaluateAt(ORUtils::SE3Pose* para) const
	{
		return new EvaluationPoint(para, this);
	}

	[[nodiscard]] int numParameters() const
	{ return (iterationType == TRACKER_ITERATION_ROTATION) ? 3 : 6; }

	virtual int F_oneLevel(float* f, ORUtils::SE3Pose* pose) = 0;

	virtual void G_oneLevel(float* gradient, float* hessian, ORUtils::SE3Pose* pose) const = 0;

	void ApplyDelta(const ORUtils::SE3Pose& para_old, const float* delta, ORUtils::SE3Pose& para_new) const;

	void TrackCamera(ITMTrackingState* trackingState, const ITMView* view) override;

	[[nodiscard]] bool requiresColourRendering() const override
	{ return true; }

	[[nodiscard]] bool requiresDepthReliability() const override
	{ return false; }

	[[nodiscard]] bool requiresPointCloudRendering() const override
	{ return true; }

	ITMColorTracker(Vector2i imgSize, TrackerIterationType* trackingRegime, int noHierarchyLevels,
	                const ITMLowLevelEngine* lowLevelEngine, MemoryDeviceType memoryType);

	~ITMColorTracker() override;
};
}
