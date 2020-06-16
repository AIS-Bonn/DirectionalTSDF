// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMViewBuilder_CPU.h"

#include "ITMLib/Engines/ViewBuilding/Shared/ITMViewBuilder_Shared.h"
#include "ORUtils/MetalContext.h"
#include "ORUtils/FileUtils.h"
#include "ITMLib/Utils/ITMTimer.h"

using namespace ITMLib;
using namespace ORUtils;

ITMViewBuilder_CPU::ITMViewBuilder_CPU(const ITMRGBDCalib& calib):ITMViewBuilder(calib) { }
ITMViewBuilder_CPU::~ITMViewBuilder_CPU(void) { }

void saveNormalImage(ITMView *view, const std::string &path)
{
	ITMUChar4Image *normalImage = new ITMUChar4Image(view->rgb->noDims, true, false);
	Vector4u *data_to =  normalImage->GetData(MEMORYDEVICE_CPU);
	const Vector4f *data_from =  view->depthNormal->GetData(MEMORYDEVICE_CPU);
	for (int i=0; i < view->depthNormal->noDims[0] * view->depthNormal->noDims[1]; i++)
	{
		data_to[i].x = static_cast<uchar>(abs(data_from[i].x) * 255);
		data_to[i].y = static_cast<uchar>(abs(data_from[i].y) * 255);
		data_to[i].z = static_cast<uchar>(abs(data_from[i].z) * 255);
		data_to[i].w = 255;
	}
	SaveImageToFile(normalImage, path.c_str());
}

void ITMViewBuilder_CPU::UpdateView(ITMView **view_ptr, ITMUChar4Image *rgbImage, ITMShortImage *rawDepthImage, bool useBilateralFilter, bool modelSensorNoise, bool storePreviousImage)
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

		if (modelSensorNoise)
		{
			(*view_ptr)->depthNormal = new ITMFloat4Image(rawDepthImage->noDims, true, false);
			(*view_ptr)->depthUncertainty = new ITMFloatImage(rawDepthImage->noDims, true, false);
		}
	}
	ITMView *view = *view_ptr;

	if (storePreviousImage)
	{
		if (!view->rgb_prev) view->rgb_prev = new ITMUChar4Image(rgbImage->noDims, true, false);
		else view->rgb_prev->SetFrom(view->rgb, MemoryBlock<Vector4u>::CPU_TO_CPU);
	}

	view->rgb->SetFrom(rgbImage, MemoryBlock<Vector4u>::CPU_TO_CPU);
	this->shortImage->SetFrom(rawDepthImage, MemoryBlock<short>::CPU_TO_CPU);

	switch (view->calib.disparityCalib.GetType())
	{
	case ITMDisparityCalib::TRAFO_KINECT:
		this->ConvertDisparityToDepth(view->depth, this->shortImage, &(view->calib.intrinsics_d), view->calib.disparityCalib.GetParams());
		break;
	case ITMDisparityCalib::TRAFO_AFFINE:
		this->ConvertDepthAffineToFloat(view->depth, this->shortImage, view->calib.disparityCalib.GetParams());
		break;
	default:
		break;
	}
	timeStats.copyImages = timer.Tock();

	if (useBilateralFilter)
	{
		timer.Tick();
		//5 steps of bilateral filtering
		this->DepthFiltering(this->floatImage, view->depth);
		this->DepthFiltering(view->depth, this->floatImage);
		this->DepthFiltering(this->floatImage, view->depth);
		this->DepthFiltering(view->depth, this->floatImage);
		this->DepthFiltering(this->floatImage, view->depth);
		view->depth->SetFrom(this->floatImage, MemoryBlock<float>::CPU_TO_CPU);
		timeStats.bilateralFilter = timer.Tock();
	}

	if (modelSensorNoise)
	{
		timer.Tick();
		this->ComputeNormalAndWeights(this->normals, view->depthUncertainty, view->depth, view->calib.intrinsics_d.projectionParamsSimple.all);
		this->NormalFiltering(view->depthNormal, this->normals);
		timeStats.normalEstimation = timer.Tock();
		saveNormalImage(view, "/tmp/normals.png");
	}
}

void ITMViewBuilder_CPU::UpdateView(ITMView **view_ptr, ITMUChar4Image *rgbImage, ITMShortImage *depthImage, bool useBilateralFilter, ITMIMUMeasurement *imuMeasurement, bool modelSensorNoise, bool storePreviousImage)
{
	if (*view_ptr == NULL)
	{
		*view_ptr = new ITMViewIMU(calib, rgbImage->noDims, depthImage->noDims, false);
		if (this->shortImage != NULL) delete this->shortImage;
		this->shortImage = new ITMShortImage(depthImage->noDims, true, false);
		if (this->floatImage != NULL) delete this->floatImage;
		this->floatImage = new ITMFloatImage(depthImage->noDims, true, false);

		if (modelSensorNoise)
		{
			(*view_ptr)->depthNormal = new ITMFloat4Image(depthImage->noDims, true, false);
			(*view_ptr)->depthUncertainty = new ITMFloatImage(depthImage->noDims, true, false);
		}
	}

	ITMViewIMU* imuView = (ITMViewIMU*)(*view_ptr);
	imuView->imu->SetFrom(imuMeasurement);

	this->UpdateView(view_ptr, rgbImage, depthImage, useBilateralFilter, modelSensorNoise, storePreviousImage);
}

void ITMViewBuilder_CPU::ConvertDisparityToDepth(ITMFloatImage *depth_out, const ITMShortImage *depth_in, const ITMIntrinsics *depthIntrinsics,
	Vector2f disparityCalibParams)
{
	Vector2i imgSize = depth_in->noDims;

	const short *d_in = depth_in->GetData(MEMORYDEVICE_CPU);
	float *d_out = depth_out->GetData(MEMORYDEVICE_CPU);

	float fx_depth = depthIntrinsics->projectionParamsSimple.fx;

	for (int y = 0; y < imgSize.y; y++) for (int x = 0; x < imgSize.x; x++)
		convertDisparityToDepth(d_out, x, y, d_in, disparityCalibParams, fx_depth, imgSize);
}

void ITMViewBuilder_CPU::ConvertDepthAffineToFloat(ITMFloatImage *depth_out, const ITMShortImage *depth_in, const Vector2f depthCalibParams)
{
	Vector2i imgSize = depth_in->noDims;

	const short *d_in = depth_in->GetData(MEMORYDEVICE_CPU);
	float *d_out = depth_out->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < imgSize.y; y++) for (int x = 0; x < imgSize.x; x++)
		convertDepthAffineToFloat(d_out, x, y, d_in, imgSize, depthCalibParams);
}

void ITMViewBuilder_CPU::DepthFiltering(ITMFloatImage *image_out, const ITMFloatImage *image_in)
{
	Vector2i imgSize = image_in->noDims;

	image_out->Clear();

	float *imout = image_out->GetData(MEMORYDEVICE_CPU);
	const float *imin = image_in->GetData(MEMORYDEVICE_CPU);

	for (int y = 2; y < imgSize.y - 2; y++) for (int x = 2; x < imgSize.x - 2; x++)
		filterDepth(imout, imin, x, y, imgSize);
}

void ITMViewBuilder_CPU::ComputeNormalAndWeights(ITMFloat4Image *normal_out, ITMFloatImage *sigmaZ_out, const ITMFloatImage *depth_in, Vector4f intrinsic)
{
	Vector2i imgDims = depth_in->noDims;

	const float *depthData_in = depth_in->GetData(MEMORYDEVICE_CPU);

	float *sigmaZData_out = sigmaZ_out->GetData(MEMORYDEVICE_CPU);
	Vector4f *normalData_out = normal_out->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < imgDims.y; y++) for (int x = 0; x < imgDims.x; x++)
		computeNormalAndWeight(depthData_in, normalData_out, sigmaZData_out, x, y, imgDims, intrinsic);
}

void ITMViewBuilder_CPU::NormalFiltering(ITMFloat4Image* normals_out, const ITMFloat4Image* normals_in)
{
	Vector2i imgDims = normals_in->noDims;

	Vector4f *n_out = normals_out->GetData(MEMORYDEVICE_CPU);
	const Vector4f *n_in = normals_in->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < imgDims.y; y++) for (int x = 0; x < imgDims.x; x++)
		filterNormals(n_out, n_in, 5, 5, x, y, imgDims);

//	for (int y = 2; y < imgDims.y - 2; y++) for (int x = 2; x < imgDims.x - 2; x++)
//		normalizeNormals(normalData, x, y);
}

