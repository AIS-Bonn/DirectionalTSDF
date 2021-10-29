// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <ORUtils/Image.h>
#include <ITMLib/Utils/ITMMath.h>

#include <stdlib.h>

namespace ITMLib
{
class ITMMesh
{
public:
	struct Triangle
	{
		Vector3f p0, p1, p2;
	};

	uint noTotalTriangles;
	uint noMaxTriangles;

	ORUtils::MemoryBlock<Triangle>* triangles;

	void Resize(size_t N)
	{
		this->noMaxTriangles = N;
		triangles->Resize(N);
	}

	ITMMesh()
	{
		noTotalTriangles = 0;
		noMaxTriangles = 1;

		triangles = new ORUtils::MemoryBlock<Triangle>(noMaxTriangles, MEMORYDEVICE_CPU);
	}

	void WriteOBJ(const char* fileName)
	{
		Triangle* triangleArray = triangles->GetData(MEMORYDEVICE_CPU);

		FILE* f = fopen(fileName, "w+");
		if (f != nullptr)
		{
			for (uint i = 0; i < noTotalTriangles; i++)
			{
				fprintf(f, "v %f %f %f\n", triangleArray[i].p0.x, triangleArray[i].p0.y, triangleArray[i].p0.z);
				fprintf(f, "v %f %f %f\n", triangleArray[i].p1.x, triangleArray[i].p1.y, triangleArray[i].p1.z);
				fprintf(f, "v %f %f %f\n", triangleArray[i].p2.x, triangleArray[i].p2.y, triangleArray[i].p2.z);
			}

			for (uint i = 0; i < noTotalTriangles; i++)
				fprintf(f, "f %d %d %d\n", i * 3 + 2 + 1, i * 3 + 1 + 1, i * 3 + 0 + 1);
			fclose(f);
		}
	}

	void WriteSTL(const char* fileName)
	{
		Triangle* triangleArray = triangles->GetData(MEMORYDEVICE_CPU);

		FILE* f = fopen(fileName, "wb+");

		if (f != nullptr)
		{
			for (int i = 0; i < 80; i++) fwrite(" ", sizeof(char), 1, f);

			fwrite(&noTotalTriangles, sizeof(int), 1, f);

			float zero = 0.0f;
			short attribute = 0;
			for (uint i = 0; i < noTotalTriangles; i++)
			{
				fwrite(&zero, sizeof(float), 1, f);
				fwrite(&zero, sizeof(float), 1, f);
				fwrite(&zero, sizeof(float), 1, f);

				fwrite(&triangleArray[i].p2.x, sizeof(float), 1, f);
				fwrite(&triangleArray[i].p2.y, sizeof(float), 1, f);
				fwrite(&triangleArray[i].p2.z, sizeof(float), 1, f);

				fwrite(&triangleArray[i].p1.x, sizeof(float), 1, f);
				fwrite(&triangleArray[i].p1.y, sizeof(float), 1, f);
				fwrite(&triangleArray[i].p1.z, sizeof(float), 1, f);

				fwrite(&triangleArray[i].p0.x, sizeof(float), 1, f);
				fwrite(&triangleArray[i].p0.y, sizeof(float), 1, f);
				fwrite(&triangleArray[i].p0.z, sizeof(float), 1, f);

				fwrite(&attribute, sizeof(short), 1, f);
			}

			fclose(f);
		}
	}

	~ITMMesh()
	{
		delete triangles;
	}

	// Suppress the default copy constructor and assignment operator
	ITMMesh(const ITMMesh&);

	ITMMesh& operator=(const ITMMesh&);
};
}
