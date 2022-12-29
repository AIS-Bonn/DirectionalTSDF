//
// Created by Malte Splietker on 28.12.22.
//

#pragma once

namespace ITMLib
{

template<typename T>
struct square : public thrust::unary_function<T, T>
{
	__host__ __device__

	T operator()(const T& x) const
	{
		return x * x;
	}
};

}