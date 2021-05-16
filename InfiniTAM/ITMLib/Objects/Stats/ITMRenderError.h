//
// Created by Malte Splietker on 11.05.21.
//

#pragma once

struct ITMRenderError
{
	ITMRenderError()
	: MAE(0), RMSE(0), icpMAE(0), icpRMSE(0) { }

	/// mean absolute error
	float MAE;

	/// root mean squared error
	float RMSE;

	/// mean absolute ICP error (using normal)
	float icpMAE;

	/// root mean squared ICP error (using normal)
	float icpRMSE;
};
