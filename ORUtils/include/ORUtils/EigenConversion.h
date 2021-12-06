//
// Created by Malte Splietker on 23.11.21.
//

#pragma once

#include <ITMLib/Utils/ITMMath.h>
#include <Eigen/Dense>

namespace ORUtils
{

template<typename T1, typename T2>
void ToEigen(const ORUtils::Vector3<T1>& in, Eigen::Matrix<T2, 3, 1>& out)
{
	for (int i = 0; i < 3; i++)
	{
		out.data()[i] = static_cast<T1>(in.v[i]);
	}
}

template<typename T1, typename T2>
void ToEigen(const ORUtils::Vector4<T1>& in, Eigen::Matrix<T2, 4, 1>& out)
{
	for (int i = 0; i < 4; i++)
	{
		out.data()[i] = static_cast<T1>(in.v[i]);
	}
}

template<typename T1, typename T2>
void ToEigen(const ORUtils::Matrix3<T1>& in, Eigen::Matrix<T2, 3, 3>& out)
{
	for (int i = 0; i < 9; i++)
	{
		out.data()[i] = static_cast<T1>(in.m[i]);
	}
}

template<typename T1, typename T2>
void ToEigen(const ORUtils::Matrix4<T1>& in, Eigen::Matrix<T2, 4, 4>& out)
{
	for (int i = 0; i < 16; i++)
	{
		out.data()[i] = static_cast<T1>(in.m[i]);
	}
}

template<typename T_Eigen, typename T_ORUtils>
T_Eigen ToEigen(const T_ORUtils& in)
{
	T_Eigen out;
	ToEigen(in, out);
	return out;
}

template<typename T1, typename T2>
void FromEigen(const Eigen::Matrix<T1, 3, 1>& in, ORUtils::Vector3<T2>& out)
{
	for (int i = 0; i < 3; i++)
	{
		out.v[i] = static_cast<T1>(in.data()[i]);
	}
}

template<typename T1, typename T2>
void FromEigen(const Eigen::Matrix<T1, 4, 1>& in, ORUtils::Vector4<T2>& out)
{
	for (int i = 0; i < 3; i++)
	{
		out.v[i] = static_cast<T1>(in.data()[i]);
	}
}

template<typename T1, typename T2>
void FromEigen(const Eigen::Matrix<T1, 3, 3>& in, ORUtils::Matrix3<T2>& out)
{
	for (int i = 0; i < 9; i++)
	{
		out.m[i] = static_cast<T1>(in.data()[i]);
	}
}

template<typename T1, typename T2>
void FromEigen(const Eigen::Matrix<T1, 4, 4>& in, ORUtils::Matrix4<T2>& out)
{
	for (int i = 0; i < 16; i++)
	{
		out.m[i] = static_cast<T1>(in.data()[i]);
	}
}

template<typename T_ORUtils, typename T_Eigen>
T_ORUtils FromEigen(const T_Eigen& in)
{
	T_ORUtils out;
	FromEigen(in, out);
	return out;
}

}