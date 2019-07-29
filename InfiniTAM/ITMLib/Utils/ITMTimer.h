//
// Created by Malte Splietker on 26.07.19.
//

#pragma once

#include <chrono>

namespace ITMLib
{

class ITMTimer
{
public:
	void Tick()
	{
		start = std::chrono::high_resolution_clock::now();
	}

	double Tock()
	{
		std::chrono::time_point<std::chrono::system_clock> end = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
	}

private:
	std::chrono::time_point<std::chrono::system_clock> start;
};

}
