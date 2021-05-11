//
// Created by Malte Splietker on 11.05.21.
//

#pragma once

struct ITMRenderError
{
	ITMRenderError()
	: min(0), max(0), average(0), variance(0) { }

	float min;
	float max;
	float average;
	float variance;
};
