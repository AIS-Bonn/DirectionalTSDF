// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include <Trackers/Shared/ITMTracker_Shared.h>
#include "ITMICPTracker_CPU.h"
#include "../Shared/ITMICPTracker_Shared.h"

using namespace ITMLib;

ITMICPTracker_CPU::ITMICPTracker_CPU(Vector2i imgSize_d, Vector2i imgSize_rgb, const Parameters& parameters,
                                     const ITMLowLevelEngine* lowLevelEngine)
	: ITMICPTracker(imgSize_d, imgSize_rgb, parameters, lowLevelEngine, MEMORYDEVICE_CPU)
{}

ITMICPTracker_CPU::~ITMICPTracker_CPU(void)
{}

int ITMICPTracker_CPU::ComputeGandH_Depth(float& f, float* nabla, float* hessian, const Matrix4f& deltaT)
{
	Vector4f* pointsMap = sceneHierarchy->GetLevel(0)->pointsMap->GetData(MEMORYDEVICE_CPU);
	Vector4f* normalsMap = sceneHierarchy->GetLevel(0)->normalsMap->GetData(MEMORYDEVICE_CPU);
	Vector4f sceneIntrinsics = sceneHierarchy->GetLevel(0)->intrinsics;
	Vector2i sceneImageSize = sceneHierarchy->GetLevel(0)->pointsMap->noDims;

	float* depth = viewHierarchy_depth->GetLevel(levelId)->data->GetData(MEMORYDEVICE_CPU);
	Vector4f viewIntrinsics = viewHierarchy_depth->GetLevel(levelId)->intrinsics;
	Vector2i viewImageSize = viewHierarchy_depth->GetLevel(levelId)->data->noDims;

	if (iterationType == TRACKER_ITERATION_NONE) return 0;

	bool shortIteration =
		(iterationType == TRACKER_ITERATION_ROTATION) || (iterationType == TRACKER_ITERATION_TRANSLATION);

	float sumHessian[6 * 6], sumNabla[6], sumF;
	int noValidPoints;
	int noPara = shortIteration ? 3 : 6, noParaSQ = shortIteration ? 3 + 2 + 1 : 6 + 5 + 4 + 3 + 2 + 1;

	noValidPoints = 0;
	sumF = 0.0f;
	memset(sumHessian, 0, sizeof(float) * noParaSQ);
	memset(sumNabla, 0, sizeof(float) * noPara);

	for (int y = 0; y < viewImageSize.y; y++)
		for (int x = 0; x < viewImageSize.x; x++)
		{
			float localHessian[6 + 5 + 4 + 3 + 2 + 1], localNabla[6], localF = 0;

			for (int i = 0; i < noPara; i++) localNabla[i] = 0.0f;
			for (int i = 0; i < noParaSQ; i++) localHessian[i] = 0.0f;

			bool isValidPoint;

			switch (iterationType)
			{
				case TRACKER_ITERATION_ROTATION:
					isValidPoint = computePerPointGH_Depth<true, true>(localNabla, localHessian, localF, x, y,
					                                                   deltaT, renderedScenePose,
					                                                   depth, viewImageSize,
					                                                   viewIntrinsics, sceneImageSize, sceneIntrinsics,
					                                                   pointsMap, normalsMap,
					                                                   distThresh[levelId]);
					break;
				case TRACKER_ITERATION_TRANSLATION:
					isValidPoint = computePerPointGH_Depth<true, false>(localNabla, localHessian, localF, x, y,
					                                                    deltaT, renderedScenePose,
					                                                    depth, viewImageSize,
					                                                    viewIntrinsics, sceneImageSize, sceneIntrinsics,
					                                                    pointsMap, normalsMap,
					                                                    distThresh[levelId]);
					break;
				case TRACKER_ITERATION_BOTH:
					isValidPoint = computePerPointGH_Depth<false, false>(localNabla, localHessian, localF, x, y,
					                                                     deltaT, renderedScenePose,
					                                                     depth, viewImageSize,
					                                                     viewIntrinsics, sceneImageSize, sceneIntrinsics,
					                                                     pointsMap, normalsMap,
					                                                     distThresh[levelId]);
					break;
				default:
					isValidPoint = false;
					break;
			}

			if (isValidPoint)
			{
				noValidPoints++;
				sumF += localF;
				for (int i = 0; i < noPara; i++) sumNabla[i] += localNabla[i];
				for (int i = 0; i < noParaSQ; i++) sumHessian[i] += localHessian[i];
			}
		}

	for (int r = 0, counter = 0; r < noPara; r++)
		for (int c = 0; c <= r; c++, counter++)
			hessian[r + c *
			            6] = sumHessian[counter];
	for (int r = 0; r < noPara; ++r) for (int c = r + 1; c < noPara; c++) hessian[r + c * 6] = hessian[c + r * 6];

	memcpy(nabla, sumNabla, noPara * sizeof(float));
	f = (noValidPoints > 100) ? sumF / noValidPoints : 1e5f;

	return noValidPoints;
}

int ITMICPTracker_CPU::ComputeGandH_RGB(float& f, float* nabla, float* hessian, const Matrix4f& delta_T)
{
	// FIXME: implement
}

void ITMICPTracker_CPU::ComputeDepthPointAndIntensity(ITMFloat4Image* points_out, ITMFloatImage* intensity_out,
                                                      const ITMFloatImage* intensity_in, const ITMFloatImage* depth_in,
                                                      const Vector4f& intrinsics_depth, const Vector4f& intrinsics_rgb,
                                                      const Matrix4f& scenePose)
{
	const Vector2i imageSize_rgb = intensity_in->noDims;
	const Vector2i imageSize_depth = depth_in->noDims; // Also the size of the projected image

	points_out->ChangeDims(imageSize_depth); // Actual reallocation should happen only once per run.
	intensity_out->ChangeDims(imageSize_depth); // Actual reallocation should happen only once per run.

	const float* depths = depth_in->GetData(MEMORYDEVICE_CPU);
	const float* intensityIn = intensity_in->GetData(MEMORYDEVICE_CPU);
	Vector4f* pointsOut = points_out->GetData(MEMORYDEVICE_CPU);
	float* intensityOut = intensity_out->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < imageSize_depth.y; y++)
		for (int x = 0; x < imageSize_depth.x; x++)
			computeDepthPointAndColour(x, y, pointsOut, intensityOut, intensityIn, depths, imageSize_rgb, imageSize_depth,
			                           intrinsics_rgb, intrinsics_depth, scenePose);
}

size_t ITMICPTracker_CPU::ComputeGandHSim3_Depth(float& f, Eigen::Matrix<EigenT, 7, 7>& H, Eigen::Matrix<EigenT, 7, 1>& g, const Matrix4f& approxInvPose)
{
	// FIXME: implement
	return 0;
}

size_t ITMICPTracker_CPU::ComputeGandHSim3_RGB(float& f, Eigen::Matrix<EigenT, 7, 7>& H, Eigen::Matrix<EigenT, 7, 1>& g, const Matrix4f& approxInvPose)
{
	// FIXME: implement
	return 0;
}
