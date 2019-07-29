//
// Created by Malte Splietker on 26.07.19.
//

#pragma once

//#include

namespace ITMLib
{

struct ITMTimeStatsBase
{
	virtual void Reset() = 0;

	virtual float GetSum() const = 0;

	virtual void Print(std::ostream &stream) const = 0;

	virtual void PrintHeader(std::ostream &stream) const = 0;

	friend std::ostream& operator<<( std::ostream& stream, const ITMTimeStatsBase& timeStats )
	{
		timeStats.Print(stream);
		return stream;
	}
};

} // namespace ITMLib
