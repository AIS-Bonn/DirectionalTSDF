// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMViewBuilder_CPU.h"

#include <Engines/ViewBuilding/Shared/ITMViewBuilder_Shared.h>
#include <Utils/ITMTimer.h>
#include <ORUtils/FileUtils.h>

using namespace ITMLib;
using namespace ORUtils;

ITMViewBuilder_CPU::ITMViewBuilder_CPU(const ITMRGBDCalib& calib) : ITMViewBuilder(calib)
{}

ITMViewBuilder_CPU::~ITMViewBuilder_CPU() = default;

void ITMViewBuilder_CPU::UpdateView(ITMView** view_ptr, ITMUChar4Image* rgbImage, ITMShortImage* rawDepthImage,
                                    bool useDepthFilter, bool useBilateralFilter, bool computeNormals)
{
	timeStats.Reset();
	ITMTimer timer;
	timer.Tick();

	if (*view_ptr == nullptr)
	{
		*view_ptr = new ITMView(calib, rgbImage->noDims, rawDepthImage->noDims, false);
		delete this->shortImage;
		this->shortImage = new ITMShortImage(rawDepthImage->noDims, true, false);
		delete this->floatImage;
		this->floatImage = new ITMFloatImage(rawDepthImage->noDims, true, false);
		delete this->normals;
		this->normals = new ITMFloat4Image(rawDepthImage->noDims, true, false);
	}
	ITMView* view = *view_ptr;

	view->rgb->SetFrom(rgbImage, ORUtils::CPU_TO_CPU);
	this->shortImage->SetFrom(rawDepthImage, ORUtils::CPU_TO_CPU);

	switch (view->calib.disparityCalib.GetType())
	{
		case ITMDisparityCalib::TRAFO_KINECT:
			this->ConvertDisparityToDepth(view->depth, this->shortImage, &(view->calib.intrinsics_d),
			                              view->calib.disparityCalib.GetParams(), useDepthFilter);
			break;
		case ITMDisparityCalib::TRAFO_AFFINE:
			this->ConvertDepthAffineToFloat(view->depth, this->shortImage, view->calib.disparityCalib.GetParams(), useDepthFilter);
			break;
		default:
			break;
	}
	timeStats.copyImages = timer.Tock();

	timer.Tick();
	this->DepthBilateralFiltering(this->floatImage, view->depth);
	timeStats.bilateralFilter = timer.Tock();
	if (useBilateralFilter)
	{
		view->depth->SetFrom(this->floatImage, ORUtils::CPU_TO_CPU);
	}

	if (computeNormals)
	{
		timer.Tick();
		this->ComputeNormalAndWeights(this->normals, this->floatImage, view->calib.intrinsics_d.projectionParamsSimple.all);
		this->NormalBilateralFiltering(view->depthNormal, this->normals);
		timeStats.normalEstimation = timer.Tock();
	}
}

void ITMViewBuilder_CPU::UpdateView(ITMView** view_ptr, ITMUChar4Image* rgbImage, ITMShortImage* depthImage,
                                    bool useDepthFilter, bool useBilateralFilter, ITMIMUMeasurement* imuMeasurement, bool computeNormals)
{
	if (*view_ptr == nullptr)
	{
		*view_ptr = new ITMViewIMU(calib, rgbImage->noDims, depthImage->noDims, false);
		if (this->shortImage != nullptr) delete this->shortImage;
		this->shortImage = new ITMShortImage(depthImage->noDims, true, false);
		if (this->floatImage != nullptr) delete this->floatImage;
		this->floatImage = new ITMFloatImage(depthImage->noDims, true, false);
	}

	ITMViewIMU* imuView = (ITMViewIMU*) (*view_ptr);
	imuView->imu->SetFrom(imuMeasurement);

	this->UpdateView(view_ptr, rgbImage, depthImage, useDepthFilter, useBilateralFilter, computeNormals);
}

void ITMViewBuilder_CPU::ConvertDisparityToDepth(ITMFloatImage* depth_out, const ITMShortImage* depth_in,
                                                 const ITMIntrinsics* depthIntrinsics,
                                                 Vector2f disparityCalibParams, bool filterDepth)
{
	Vector2i imgSize = depth_in->noDims;

	const short* d_in = depth_in->GetData(MEMORYDEVICE_CPU);
	float* d_out = depth_out->GetData(MEMORYDEVICE_CPU);

	float fx_depth = depthIntrinsics->projectionParamsSimple.fx;

	for (int y = 0; y < imgSize.y; y++)
		for (int x = 0; x < imgSize.x; x++)
			convertDisparityToDepth(d_out, x, y, d_in, disparityCalibParams, fx_depth, imgSize, filterDepth);
}

void ITMViewBuilder_CPU::ConvertDepthAffineToFloat(ITMFloatImage* depth_out, const ITMShortImage* depth_in,
                                                   const Vector2f depthCalibParams, bool filterDepth)
{
	Vector2i imgSize = depth_in->noDims;

	const short* d_in = depth_in->GetData(MEMORYDEVICE_CPU);
	float* d_out = depth_out->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < imgSize.y; y++)
		for (int x = 0; x < imgSize.x; x++)
			convertDepthAffineToFloat(d_out, x, y, d_in, imgSize, depthCalibParams, filterDepth);
}

void ITMViewBuilder_CPU::DepthBilateralFiltering(ITMFloatImage* image_out, const ITMFloatImage* image_in)
{
	Vector2i imgSize = image_in->noDims;

	image_out->Clear();

	float* imout = image_out->GetData(MEMORYDEVICE_CPU);
	const float* imin = image_in->GetData(MEMORYDEVICE_CPU);

	for (int y = 2; y < imgSize.y - 2; y++)
		for (int x = 2; x < imgSize.x - 2; x++)
			filterDepthBilateral(imout, imin, 5.0, 0.025, x, y, imgSize);
}

void ITMViewBuilder_CPU::ComputeNormalAndWeights(ITMFloat4Image* normal_out, const ITMFloatImage* depth_in,
                                                 Vector4f intrinsic)
{
	Vector2i imgDims = depth_in->noDims;

	const float* depthData_in = depth_in->GetData(MEMORYDEVICE_CPU);

	Vector4f* normalData_out = normal_out->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < imgDims.y; y++)
		for (int x = 0; x < imgDims.x; x++)
			computeNormalAndWeight(depthData_in, normalData_out, x, y, imgDims, intrinsic);
}

void ITMViewBuilder_CPU::NormalBilateralFiltering(ITMFloat4Image* normals_out, const ITMFloat4Image* normals_in)
{
	Vector2i imgDims = normals_in->noDims;

	Vector4f* n_out = normals_out->GetData(MEMORYDEVICE_CPU);
	const Vector4f* n_in = normals_in->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < imgDims.y; y++)
		for (int x = 0; x < imgDims.x; x++)
			filterNormalsBilateral(n_out, n_in, 2.5, 5, x, y, imgDims);

//	for (int y = 2; y < imgDims.y - 2; y++) for (int x = 2; x < imgDims.x - 2; x++)
//		normalizeNormals(normalData, x, y);
}

