//
// Created by Malte Splietker on 01.10.19.
//

#pragma once

struct iequal
{
	bool operator()(int c1, int c2) const
	{
		return std::toupper(c1) == std::toupper(c2);
	}
};

/**
 * Case insensitive string comparison
 * @param str1
 * @param str2
 * @return true if equal (ignoring case)
 */
inline bool iequals(const std::string& str1, const std::string& str2)
{
	return std::equal(str1.begin(), str1.end(), str2.begin(), iequal());
}

