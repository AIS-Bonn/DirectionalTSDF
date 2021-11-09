//
// Created by Malte Splietker on 18.10.21.
//

#include "PointCloudSourceEngine.h"

#include <tinyply/source/tinyply.h>

using namespace ITMLib;

inline std::vector<uint8_t> read_file_binary(const std::string & pathToFile)
{
	std::ifstream file(pathToFile, std::ios::binary);
	std::vector<uint8_t> fileBufferBytes;

	if (file.is_open())
	{
		file.seekg(0, std::ios::end);
		size_t sizeBytes = file.tellg();
		file.seekg(0, std::ios::beg);
		fileBufferBytes.resize(sizeBytes);
		if (file.read((char*)fileBufferBytes.data(), sizeBytes)) return fileBufferBytes;
	}
	else throw std::runtime_error("could not open binary ifstream to path " + pathToFile);
	return fileBufferBytes;
}

struct memory_buffer : public std::streambuf
{
	char * p_start {nullptr};
	char * p_end {nullptr};
	size_t size;

	memory_buffer(char const * first_elem, size_t size)
		: p_start(const_cast<char*>(first_elem)), p_end(p_start + size), size(size)
	{
		setg(p_start, p_start, p_end);
	}

	pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override
	{
		if (dir == std::ios_base::cur) gbump(static_cast<int>(off));
		else setg(p_start, (dir == std::ios_base::beg ? p_start : p_end) + off, p_end);
		return gptr() - p_start;
	}

	pos_type seekpos(pos_type pos, std::ios_base::openmode which) override
	{
		return seekoff(pos, std::ios_base::beg, which);
	}
};

struct memory_stream : virtual memory_buffer, public std::istream
{
	memory_stream(char const * first_elem, size_t size)
		: memory_buffer(first_elem, size), std::istream(static_cast<std::streambuf*>(this)) {}
};

namespace InputSource
{

PointCloudSourceEngine::PointCloudSourceEngine(const std::string& directory)
	: currentPointCloudIdx(0)
{
	currentPointCloud = new ITMLib::ITMPointCloud(Vector2i(1, 1), MEMORYDEVICE_CPU);
	std::copy(std::filesystem::directory_iterator(directory), std::filesystem::directory_iterator(),
	          std::back_inserter(pointCloudPaths));
	std::sort(pointCloudPaths.begin(), pointCloudPaths.end());
}

void PointCloudSourceEngine::ReadPLY(const std::string& filepath)
{
	std::unique_ptr<std::istream> file_stream;
	std::vector<uint8_t> byte_buffer;

	try
	{
		byte_buffer = read_file_binary(filepath);
		file_stream.reset(new memory_stream((char*) byte_buffer.data(), byte_buffer.size()));

		if (!file_stream || file_stream->fail()) throw std::runtime_error("file_stream failed to open " + filepath);

		file_stream->seekg(0, std::ios::end);
		const float size_mb = file_stream->tellg() * float(1e-6);
		file_stream->seekg(0, std::ios::beg);

		tinyply::PlyFile file;
		file.parse_header(*file_stream);

		std::shared_ptr<tinyply::PlyData> points, normals, colors;
		try
		{
			points = file.request_properties_from_element("vertex", {"x", "y", "z", "intensity"});
		}
		catch (const std::exception& e)
		{
			std::cerr << "tinyply exception: " << e.what() << std::endl;
			return;
		}

		// optional normals
		try
		{
			normals = file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
		}
		catch (const std::exception& e)
		{}

		// optional color
		try
		{
			colors = file.request_properties_from_element("vertex", {"r", "g", "b", "a"});
		}
		catch (const std::exception& e)
		{}

		file.read(*file_stream);

		const size_t numVerticesBytes = points->buffer.size_bytes();
		currentPointCloud->Resize(Vector2i(1024, 64));
		memcpy(currentPointCloud->locations->GetData(MEMORYDEVICE_CPU), points->buffer.get(), numVerticesBytes);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
	}
}

void PointCloudSourceEngine::getPointCloud(ITMLib::ITMPointCloud* pointCloud)
{
	pointCloud->locations = currentPointCloud->locations;
	pointCloud->normals = currentPointCloud->normals;
	pointCloud->colours = currentPointCloud->colours;
}

};