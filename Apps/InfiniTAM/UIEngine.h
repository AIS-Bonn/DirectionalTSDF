// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include <InputSource/ImageSourceEngine.h>
#include <InputSource/IMUSourceEngine.h>
#include <InputSource/FFMPEGWriter.h>
#include <ITMLib/Core/ITMMainEngine.h>
#include <ITMLib/Utils/ITMLibSettings.h>
#include <ITMLib/Engines/ITMLoggingEngine.h>
#include <ORUtils/FileUtils.h>
#include <ORUtils/NVTimer.h>

#include <chrono>
#include <vector>
#include <Apps/Utils/CLIUtils.h>
#include <Apps/AppEngine/AppEngine.h>

typedef std::chrono::system_clock Clock;

namespace InfiniTAM
{
	namespace Engine
	{
		class UIEngine : public AppEngine
		{
			static UIEngine* instance;

			enum MainLoopAction
			{
				PROCESS_PAUSED, PROCESS_FRAME, PROCESS_VIDEO, EXIT, SAVE_TO_DISK
			}mainLoopAction;

			struct UIColourMode {
				const char *name;
				ITMLib::ITMMainEngine::GetImageType type;
				UIColourMode(const char *_name, ITMLib::ITMMainEngine::GetImageType _type)
				 : name(_name), type(_type)
				{}
			};
			std::vector<UIColourMode> colourModes_main, colourModes_freeview;
			int currentColourMode;

		private: // For UI layout
			static const int NUM_WIN = 3;
			Vector4f winReg[NUM_WIN]; // (x1, y1, x2, y2)
			Vector2i winSize;
			uint textureId[NUM_WIN];
			ITMUChar4Image *outImage[NUM_WIN];
			ITMLib::ITMMainEngine::GetImageType outImageType[NUM_WIN];

			bool freeviewActive;
			bool integrationActive;
			bool helpActive;
			bool renderAxesActive;
			ORUtils::SE3Pose freeviewPose;
			ITMLib::ITMIntrinsics freeviewIntrinsics;

			int mouseState;
			Vector2i mouseLastClick;
			bool mouseWarped; // To avoid the extra motion generated by glutWarpPointer

			bool isRecording;
			InputSource::FFMPEGWriter *rgbVideoWriter;
			InputSource::FFMPEGWriter *depthVideoWriter;
			InputSource::FFMPEGWriter *outputVideoWriter;

			void _initialise(int argc, char** argv, AppData* appData, ITMLib::ITMMainEngine *mainEngine) override;
			bool _processFrame() override;

			void printPixelInformation(int x, int y);

		public:
			static UIEngine* Instance(void) {
				if (instance == NULL) instance = new UIEngine();
				return instance;
			}

			static void glutDisplayFunction();
			static void glutReshape(int w, int h);
			static void glutIdleFunction();
			static void glutKeyFunction(unsigned char key, int x, int y);
			static void glutKeyUpFunction(unsigned char key, int x, int y);
			static void glutMouseButtonFunction(int button, int state, int x, int y);
			static void glutMouseMoveFunction(int x, int y);
			static void glutMouseWheelFunction(int button, int dir, int x, int y);

			static void displayAxes();
			static void displayHelp();

			/** Check if programm is stuck (e.g. infinit-loop in CUDA code) and exit, if necessary */
			static void checkStuck(int value);

			const Vector2i & getWindowSize(void) const
			{ return winSize; }

			float processedTime;
			int trackingResult;
			bool needsRefresh;
			ITMUChar4Image *saveImage;

			bool keysPressed[256];
			Clock::time_point lastUpdate;

			void Shutdown();

			void Run();

			void GetScreenshot(ITMUChar4Image *dest) const;
			void SaveScreenshot(const char *filename) const;
		};
	}
}
