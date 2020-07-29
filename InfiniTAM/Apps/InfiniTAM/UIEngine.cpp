// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "UIEngine.h"

#include <string.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#ifdef FREEGLUT
#include <GL/freeglut.h>
#include <ITMLib/Engines/ViewBuilding/ITMViewBuilderFactory.h>

#else
#if (!defined USING_CMAKE) && (defined _MSC_VER)
#pragma comment(lib, "glut64")
#endif
#endif

#include "../../ITMLib/ITMLibDefines.h"
#include "../../ITMLib/Core/ITMBasicEngine.h"
#include "../../ITMLib/Core/ITMBasicSurfelEngine.h"
#include "../../ITMLib/Core/ITMMultiEngine.h"

#include "../../ORUtils/FileUtils.h"
#include "../../InputSource/FFMPEGWriter.h"

using namespace InfiniTAM::Engine;
using namespace InputSource;
using namespace ITMLib;

UIEngine* UIEngine::instance;

static void safe_glutBitmapString(void *font, const char *str)
{
	size_t len = strlen(str);
	for (size_t x = 0; x < len; ++x) {
		glutBitmapCharacter(font, str[x]);
	}
}

void UIEngine::glutReshape(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	float r = float(w) / h;
	glFrustum(-r, r, -1.0, 1.0, 1.5, 20.0); // Set frustum projection for coordinate axes
	glMatrixMode(GL_MODELVIEW);
}

void UIEngine::displayHelp()
{
//	UIEngine *uiEngine = UIEngine::Instance();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	{
		glLoadIdentity();

		const static unsigned char helpString[] = "Fusion\n"
		                                          "n   - process single frame\n"
		                                          "b   - process continuously\n"
		                                          "t   - activate/deactivate integration\n"
		                                          "r   - reset scene\n"
		                                          "Esc - exit\n"
		                                          "\n"
		                                          "Display Options\n"
		                                          "f   - activate/deactivate free view\n"
		                                          "]/[ - change local map (MultiEngine only)\n"
		                                          "c   - change color mode\n"
		                                          "x   - change normal interpolation (SDF/image points)\n"
		                                          "o   - display/hide world coordinate axes\n"
		                                          "h/? - display/hide help\n"
		                                          "\n"
		                                          "File Operations\n"
		                                          "i   - start/stop image recording\n"
		                                          "v   - start/stop video recording\n"
		                                          "e   - render and store per-pose error images\n"
		                                          "m   - save scene mesh\n"
		                                          "r   - reset scene\n"
		                                          "k   - save scene to disk\n"
		                                          "l   - load scene from disk\n";


		glLineWidth(2.5);
		glTranslatef(-0.95, 0.95f, 0);
		glScalef(0.00015,0.00025,1);
		glColor3f(1.0f, 0.0f, 0.0f);
		glutStrokeString(GLUT_STROKE_MONO_ROMAN, helpString);
	}
	glPopMatrix();

}

void UIEngine::displayAxes()
{
	UIEngine *uiEngine = UIEngine::Instance();

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	{
		glLoadIdentity();

		/// Compute view matrix
		Matrix4f T_CW = uiEngine->mainEngine->GetTrackingState()->pose_d->GetM();
		if (uiEngine->freeviewActive)
		{
			T_CW = uiEngine->freeviewPose.GetM();
		}
		// GL matrices are column-major
		GLfloat R_CW_gl[16] = {
			T_CW.m00, T_CW.m01, T_CW.m02, 0, // first column
			T_CW.m10, T_CW.m11, T_CW.m12, 0, // second column
			T_CW.m20, T_CW.m21, T_CW.m22, 0,
			0, 0, 0, 1
		};

		// 3) Translate frame, s.t. it is in front of the camera (i.e. visible)
		glTranslated(0.5, -1.2, -3);

		// 2) Transform from InfiniTAM camera to GL camera
		// InfiniTAM              OpenGL
		//   z                    y
		//  /                     |
		// +----x                 +----x
		// |                     /
		// y                    z
		glRotatef(180, 1, 0, 0);

		// 1) Transform axes from world to InfiniTAM camera
		glMultMatrixf(R_CW_gl);


		/// Draw the actual axes
		const Vector3f axes[3] {Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(0, 0, 1)};
		const Vector3f colors[3] = {Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(0, 0, 1)};
		const char* axesLabels[3]{"x", "y", "z"};
		const float axesScale = 0.5;

		glLineWidth(5.0);
		glClear(GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		for (int i = 0; i < 3; i++)
		{
			glColor3f(colors[i].x, colors[i].y, colors[i].z);
			glBegin(GL_LINES);
			Vector3f vert = axesScale * axes[i];
			glVertex3f(0, 0, 0);
			glVertex3f(vert.x, vert.y, vert.z);
			glEnd();
			Vector3f vert2 = (axesScale + 0.05) * axes[i];
			glRasterPos3f(vert2.x, vert2.y, vert2.z);
			safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, axesLabels[i]);
		}
		glDisable(GL_DEPTH_TEST);
	}
	glPopMatrix();
}

void UIEngine::glutDisplayFunction()
{
	UIEngine *uiEngine = UIEngine::Instance();

	// get updated images from processing thread
	uiEngine->mainEngine->GetImage(uiEngine->outImage[0], uiEngine->outImageType[0], &uiEngine->freeviewPose, &uiEngine->freeviewIntrinsics, uiEngine->normalsFromSDF);

	for (int w = 1; w < NUM_WIN; w++) uiEngine->mainEngine->GetImage(uiEngine->outImage[w], uiEngine->outImageType[w], nullptr, nullptr, uiEngine->normalsFromSDF);

	// do the actual drawing
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0f, 1.0f, 1.0f);
	glEnable(GL_TEXTURE_2D);

	ITMUChar4Image** showImgs = uiEngine->outImage;
	Vector4f *winReg = uiEngine->winReg;
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	{
		glLoadIdentity();
		glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		{
			glEnable(GL_TEXTURE_2D);
			for (int w = 0; w < NUM_WIN; w++) {// Draw each sub window
				if (uiEngine->outImageType[w] == ITMMainEngine::InfiniTAM_IMAGE_UNKNOWN) continue;
				glBindTexture(GL_TEXTURE_2D, uiEngine->textureId[w]);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, showImgs[w]->noDims.x, showImgs[w]->noDims.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, showImgs[w]->GetData(MEMORYDEVICE_CPU));
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glBegin(GL_QUADS); {
					glTexCoord2f(0, 1); glVertex2f(winReg[w][0], winReg[w][1]); // glVertex2f(0, 0);
					glTexCoord2f(1, 1); glVertex2f(winReg[w][2], winReg[w][1]); // glVertex2f(1, 0);
					glTexCoord2f(1, 0); glVertex2f(winReg[w][2], winReg[w][3]); // glVertex2f(1, 1);
					glTexCoord2f(0, 0); glVertex2f(winReg[w][0], winReg[w][3]); // glVertex2f(0, 1);
				}
				glEnd();
			}
			glDisable(GL_TEXTURE_2D);
		}
		glPopMatrix();
	}
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	{
		glLoadIdentity();

		switch (uiEngine->trackingResult)
		{
			case 0:
				glColor3f(1.0f, 0.0f, 0.0f);
				break; // failure
			case 1:
				glColor3f(1.0f, 1.0f, 0.0f);
				break; // poor
			case 2:
				glColor3f(0.0f, 1.0f, 0.0f);
				break; // good
			default:
				glColor3f(1.0f, 1.0f, 1.0f);
				break; // relocalising
		}

		glRasterPos2f(0.9, -0.962f);

		char str[200];
		sprintf(str, "%04.2lf", uiEngine->processedTime);
		safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const char*) str);

		glColor3f(1.0f, 1.0f, 1.0f);
		glRasterPos2f(0.65, -0.962f);
		sprintf(str, "Frame: %i", uiEngine->currentFrameNo);
		safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const char*) str);


		glColor3f(1.0f, 0.0f, 0.0f);
		glRasterPos2f(-0.98f, -0.95f);

		sprintf(str,
		        "Esc: exit \t h/?: help \t free view: %s \t colour mode: %s \t normals: %s \t fusion: %s",
		        uiEngine->freeviewActive ?  "on" : "off",
		        uiEngine->colourModes_main[uiEngine->currentColourMode].name,
		        uiEngine->normalsFromSDF ? "from SDF" : "from points",
		        uiEngine->integrationActive ? "on" : "off");
		safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const char*) str);
	}
	glPopMatrix();

	if (uiEngine->renderAxesActive)
		displayAxes();

	if (uiEngine->helpActive)
		displayHelp();

	glutSwapBuffers();
	uiEngine->needsRefresh = false;
}

void UIEngine::glutIdleFunction()
{
	UIEngine *uiEngine = UIEngine::Instance();

	Clock::time_point now = Clock::now();
	double deltaT = std::chrono::duration<double>(now - uiEngine->lastUpdate).count();

	switch (uiEngine->mainLoopAction)
	{
	case PROCESS_FRAME:
		uiEngine->ProcessFrame(); uiEngine->processedFrameNo++;
		uiEngine->mainLoopAction = PROCESS_PAUSED;
		uiEngine->needsRefresh = true;
		break;
	case PROCESS_VIDEO:
		uiEngine->ProcessFrame(); uiEngine->processedFrameNo++;
		uiEngine->needsRefresh = true;
		break;
		//case SAVE_TO_DISK:
		//	if (!uiEngine->actionDone)
		//	{
		//		char outFile[255];

		//		ITMUChar4Image *saveImage = uiEngine->saveImage;

		//		glReadBuffer(GL_BACK);
		//		glReadPixels(0, 0, saveImage->noDims.x, saveImage->noDims.x, GL_RGBA, GL_UNSIGNED_BYTE, (unsigned char*)saveImage->GetData(false));
		//		sprintf(outFile, "%s/out_%05d.ppm", uiEngine->outFolder, uiEngine->processedFrameNo);

		//		SaveImageToFile(saveImage, outFile, true);

		//		uiEngine->actionDone = true;
		//	}
		//	break;
	case EXIT:
#ifdef FREEGLUT
		glutLeaveMainLoop();
#else
		exit(0);
#endif
		break;
	case PROCESS_PAUSED:
	default:
		break;
	}

	if (uiEngine->freeviewActive)
	{
		Vector3f translation(0, 0, 0);
		if (uiEngine->keysPressed[static_cast<size_t>('w')])
			translation += Vector3f(0, 0, -1);
		if (uiEngine->keysPressed[static_cast<size_t>('s')])
			translation += Vector3f(0, 0, 1);
		if (uiEngine->keysPressed[static_cast<size_t>('a')])
			translation += Vector3f(1, 0, 0);
		if (uiEngine->keysPressed[static_cast<size_t>('d')])
			translation += Vector3f(-1, 0, 0);
		if (ORUtils::length(translation) > 0)
		{
			translation = translation.normalised();
			uiEngine->freeviewPose.SetT(uiEngine->freeviewPose.GetT() + deltaT * translation);
			uiEngine->needsRefresh = true;
		}
	}

	uiEngine->lastUpdate = Clock::now();

	if (uiEngine->needsRefresh) {
		glutPostRedisplay();
	}
}

void UIEngine::glutKeyFunction(unsigned char key, int x, int y)
{
	UIEngine* uiEngine = UIEngine::Instance();
	uiEngine->keysPressed[key] = true;
}

void UIEngine::glutKeyUpFunction(unsigned char key, int x, int y)
{
	UIEngine *uiEngine = UIEngine::Instance();
	uiEngine->keysPressed[key] = false;

	switch (key)
	{
	case 'n':
		printf("processing one frame ...\n");
		uiEngine->mainLoopAction = UIEngine::PROCESS_FRAME;
		break;
	case 'b':
		printf("processing input source ...\n");
		uiEngine->mainLoopAction = UIEngine::PROCESS_VIDEO;
		break;
	case 'i':
		if (uiEngine->isRecording)
		{
			printf("stopped recoding disk ...\n");
			uiEngine->isRecording = false;
		}
		else
		{
			printf("started recoding disk ...\n");
			uiEngine->currentFrameNo = 0;
			uiEngine->isRecording = true;
		}
		break;
	case 'v':
		if ((uiEngine->rgbVideoWriter != nullptr) || (uiEngine->depthVideoWriter != nullptr))
		{
			printf("stop recoding video\n");
			delete uiEngine->rgbVideoWriter;
			delete uiEngine->depthVideoWriter;
			uiEngine->rgbVideoWriter = nullptr;
			uiEngine->depthVideoWriter = nullptr;
		}
		else
		{
			printf("start recoding video\n");
			uiEngine->rgbVideoWriter = new FFMPEGWriter();
			uiEngine->depthVideoWriter = new FFMPEGWriter();
		}
		break;
	case 27: // esc key
		printf("exiting ...\n");
		uiEngine->mainLoopAction = UIEngine::EXIT;
		break;
	case 'f':
		uiEngine->currentColourMode = 0;
		if (uiEngine->freeviewActive)
		{
			uiEngine->outImageType[0] = ITMMainEngine::InfiniTAM_IMAGE_SCENERAYCAST;
			uiEngine->outImageType[1] = ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_DEPTH;

			uiEngine->freeviewActive = false;
		}
		else
		{
			uiEngine->outImageType[0] = ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_SHADED;
			uiEngine->outImageType[1] = ITMMainEngine::InfiniTAM_IMAGE_SCENERAYCAST;

			uiEngine->freeviewPose.SetFrom(uiEngine->mainEngine->GetTrackingState()->pose_d);
			if (uiEngine->mainEngine->GetView() != nullptr) {
				uiEngine->freeviewIntrinsics = uiEngine->mainEngine->GetView()->calib.intrinsics_d;
				uiEngine->outImage[0]->ChangeDims(uiEngine->mainEngine->GetView()->depth->noDims);
			}

			ITMMultiEngine<ITMVoxel, ITMVoxelIndex> *multiEngine = dynamic_cast<ITMMultiEngine<ITMVoxel, ITMVoxelIndex>*>(uiEngine->mainEngine);
			if (multiEngine != nullptr)
			{
				int idx = multiEngine->findPrimaryLocalMapIdx();
				if (idx < 0) idx = 0;
				multiEngine->setFreeviewLocalMapIdx(idx);
			}

			uiEngine->freeviewActive = true;
		}
		uiEngine->needsRefresh = true;
		break;
	case '1': case '2': case '3': case '4': case '5':
		uiEngine->currentColourMode = (key - '1') % (uiEngine->freeviewActive ? 4 : 5);
		uiEngine->needsRefresh = true;
		break;
	case 'c':
		uiEngine->currentColourMode++;
		if (((uiEngine->freeviewActive) && ((unsigned)uiEngine->currentColourMode >= uiEngine->colourModes_freeview.size())) ||
			((!uiEngine->freeviewActive) && ((unsigned)uiEngine->currentColourMode >= uiEngine->colourModes_main.size())))
			uiEngine->currentColourMode = 0;
		uiEngine->needsRefresh = true;
		break;
	case 'x':
		uiEngine->normalsFromSDF = !uiEngine->normalsFromSDF;
		uiEngine->needsRefresh = true;
		break;
	case 't':
	{
		uiEngine->integrationActive = !uiEngine->integrationActive;

		ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *basicEngine = dynamic_cast<ITMBasicEngine<ITMVoxel, ITMVoxelIndex>*>(uiEngine->mainEngine);
		if (basicEngine != nullptr)
		{
			if (uiEngine->integrationActive) basicEngine->turnOnIntegration();
			else basicEngine->turnOffIntegration();
		}

		ITMBasicSurfelEngine<ITMSurfelT> *basicSurfelEngine = dynamic_cast<ITMBasicSurfelEngine<ITMSurfelT>*>(uiEngine->mainEngine);
		if (basicSurfelEngine != nullptr)
		{
			if (uiEngine->integrationActive) basicSurfelEngine->turnOnIntegration();
			else basicSurfelEngine->turnOffIntegration();
		}
	}
	break;
	case 'e':
		printf("Collecting ICP Error Images ... ");
		uiEngine->CollectICPErrorImages();
		printf("done\n");
		break;
	case 'm':
	{
		printf("saving scene to model ... ");
		uiEngine->mainEngine->SaveSceneToMesh("mesh.stl");
		printf("done\n");
	}
	break;
	case 'r':
	{
		ITMBasicEngine<ITMVoxel, ITMVoxelIndex> *basicEngine = dynamic_cast<ITMBasicEngine<ITMVoxel, ITMVoxelIndex>*>(uiEngine->mainEngine);
		if (basicEngine != nullptr) basicEngine->resetAll();

		ITMBasicSurfelEngine<ITMSurfelT> *basicSurfelEngine = dynamic_cast<ITMBasicSurfelEngine<ITMSurfelT>*>(uiEngine->mainEngine);
		if (basicSurfelEngine != nullptr) basicSurfelEngine->resetAll();
	}
	break;
	case 'k':
	{
		printf("saving scene to disk ... ");

		try
		{
			uiEngine->mainEngine->SaveToFile();
			printf("done\n");
		}
		catch (const std::runtime_error &e)
		{
			printf("failed: %s\n", e.what());
		}
	}
	break;
	case 'l':
	{
		printf("loading scene from disk ... ");

		try
		{
			uiEngine->mainEngine->LoadFromFile();
			printf("done\n");
		}
		catch (const std::runtime_error &e)
		{
			printf("failed: %s\n", e.what());
		}
	}
	break;
	case '[':
	case ']':
	{
		ITMMultiEngine<ITMVoxel, ITMVoxelIndex> *multiEngine = dynamic_cast<ITMMultiEngine<ITMVoxel, ITMVoxelIndex>*>(uiEngine->mainEngine);
		if (multiEngine != nullptr)
		{
			int idx = multiEngine->getFreeviewLocalMapIdx();
			if (key == '[') idx--;
			else idx++;
			multiEngine->changeFreeviewLocalMapIdx(&(uiEngine->freeviewPose), idx);
			uiEngine->needsRefresh = true;
		}
	}
	break;
	case 'o':
		uiEngine->renderAxesActive = !uiEngine->renderAxesActive;
		glutPostRedisplay();
	break;
	case 'h':
	case '?':
		uiEngine->helpActive = !uiEngine->helpActive;
		glutPostRedisplay();
	default:
		break;
	}

	if (uiEngine->freeviewActive) uiEngine->outImageType[0] = uiEngine->colourModes_freeview[uiEngine->currentColourMode].type;
	else uiEngine->outImageType[0] = uiEngine->colourModes_main[uiEngine->currentColourMode].type;
}

void UIEngine::glutMouseButtonFunction(int button, int state, int x, int y)
{
	UIEngine *uiEngine = UIEngine::Instance();

	if (state == GLUT_DOWN)
	{
		switch (button)
		{
		case GLUT_LEFT_BUTTON: uiEngine->mouseState = 1; break;
		case GLUT_MIDDLE_BUTTON: uiEngine->mouseState = 3; break;
		case GLUT_RIGHT_BUTTON: uiEngine->mouseState = 2; break;
		default: break;
		}
		uiEngine->mouseLastClick.x = x;
		uiEngine->mouseLastClick.y = y;

		glutSetCursor(GLUT_CURSOR_NONE);
	}
	else if (state == GLUT_UP && !uiEngine->mouseWarped)
	{
		uiEngine->mouseState = 0;
		glutSetCursor(GLUT_CURSOR_INHERIT);
	}
}

static inline Matrix3f createRotation(const Vector3f & _axis, float angle)
{
	Vector3f axis = normalize(_axis);
	float si = sinf(angle);
	float co = cosf(angle);

	Matrix3f ret;
	ret.setIdentity();

	ret *= co;
	for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) ret.at(c, r) += (1.0f - co) * axis[c] * axis[r];

	Matrix3f skewmat;
	skewmat.setZeros();
	skewmat.at(1, 0) = -axis.z;
	skewmat.at(0, 1) = axis.z;
	skewmat.at(2, 0) = axis.y;
	skewmat.at(0, 2) = -axis.y;
	skewmat.at(2, 1) = axis.x;
	skewmat.at(1, 2) = -axis.x;
	skewmat *= si;
	ret += skewmat;

	return ret;
}

void UIEngine::glutMouseMoveFunction(int x, int y)
{
	UIEngine *uiEngine = UIEngine::Instance();

	if (uiEngine->mouseWarped)
	{
		uiEngine->mouseWarped = false;
		return;
	}

	if (!uiEngine->freeviewActive || uiEngine->mouseState == 0) return;

	Vector2i movement;
	movement.x = x - uiEngine->mouseLastClick.x;
	movement.y = y - uiEngine->mouseLastClick.y;

	Vector2i realWinSize(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
	// Does not work if the window is smaller than 40x40
	Vector2i activeWinTopLeft(20, 20);
	Vector2i activeWinBottomRight(realWinSize.width - 20, realWinSize.height - 20);
	Vector2i activeWinSize(realWinSize.width - 40, realWinSize.height - 40);

	bool warpNeeded = false;

	if (x < activeWinTopLeft.x)
	{
		x += activeWinSize.x;
		warpNeeded = true;
	}
	else if (x >= activeWinBottomRight.x)
	{
		x -= activeWinSize.x;
		warpNeeded = true;
	}

	if (y < activeWinTopLeft.y)
	{
		y += activeWinSize.y;
		warpNeeded = true;
	}
	else if (y >= activeWinBottomRight.y)
	{
		y -= activeWinSize.y;
		warpNeeded = true;
	}

	if (warpNeeded)
	{
		glutWarpPointer(x, y);
		uiEngine->mouseWarped = true;
	}

	uiEngine->mouseLastClick.x = x;
	uiEngine->mouseLastClick.y = y;

	if ((movement.x == 0) && (movement.y == 0)) return;

	static const float scale_rotation = 0.005f;
	static const float scale_translation = 0.0025f;

	switch (uiEngine->mouseState)
	{
	case 1:
	{
		// middle button: pitch and yaw rotation
		Vector3f axis((float)-movement.y, (float)-movement.x, 0.0f);
		float angle = scale_rotation * sqrt((float)(movement.x * movement.x + movement.y*movement.y));
		Matrix3f rot = createRotation(axis, angle);
		uiEngine->freeviewPose.SetRT(rot * uiEngine->freeviewPose.GetR(), rot * uiEngine->freeviewPose.GetT());
		uiEngine->freeviewPose.Coerce();
		uiEngine->needsRefresh = true;
		break;
	}
	case 2:
	{
		// right button: translation in x and y direction
		uiEngine->freeviewPose.SetT(uiEngine->freeviewPose.GetT() + scale_translation * Vector3f((float)movement.x, (float)movement.y, 0.0f));
		uiEngine->needsRefresh = true;
		break;
	}
	case 3:
	{
		// middle button: pitch and roll rotation
		Vector3f axis((float)-movement.y, 0.0f, (float)-movement.x);
		float angle = scale_rotation * sqrt((float)(movement.x * movement.x + movement.y*movement.y));
		Matrix3f rot = createRotation(axis, angle);
		uiEngine->freeviewPose.SetRT(rot * uiEngine->freeviewPose.GetR(), rot * uiEngine->freeviewPose.GetT());
		uiEngine->freeviewPose.Coerce();
		uiEngine->needsRefresh = true;
		break;
	}
	default: break;
	}
}

void UIEngine::glutMouseWheelFunction(int button, int dir, int x, int y)
{
	UIEngine *uiEngine = UIEngine::Instance();

	static const float scale_translation = 0.05f;

	uiEngine->freeviewPose.SetT(uiEngine->freeviewPose.GetT() + scale_translation * Vector3f(0.0f, 0.0f, (dir > 0) ? -1.0f : 1.0f));
	uiEngine->needsRefresh = true;
}

void UIEngine::Initialise(int & argc, char** argv, AppData* appData, ITMMainEngine *mainEngine)
{
	this->appData = appData;
	this->freeviewActive = false;
	this->integrationActive = true;
	this->helpActive = false;
	this->renderAxesActive = true;
	this->currentColourMode = 0;
	this->normalsFromSDF = false;
	memset(keysPressed, false, sizeof(keysPressed));
	this->colourModes_main.push_back(UIColourMode("shaded greyscale", ITMMainEngine::InfiniTAM_IMAGE_SCENERAYCAST));
	this->colourModes_main.push_back(UIColourMode("integrated colours", ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_VOLUME));
	this->colourModes_main.push_back(UIColourMode("surface normals", ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_NORMAL));
	this->colourModes_main.push_back(UIColourMode("confidence", ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_CONFIDENCE));
	this->colourModes_main.push_back(UIColourMode("icp error", ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_ICP_ERROR));
	this->colourModes_freeview.push_back(UIColourMode("shaded greyscale", ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_SHADED));
	this->colourModes_freeview.push_back(UIColourMode("integrated colours", ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME));
	this->colourModes_freeview.push_back(UIColourMode("surface normals", ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL));
	this->colourModes_freeview.push_back(UIColourMode("confidence", ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_CONFIDENCE));

	this->mainEngine = mainEngine;
	{
		size_t len = appData->outputDirectory.size();
		this->outFolder = new char[len + 1];
		strcpy(this->outFolder, appData->outputDirectory.c_str());
	}

	this->statisticsEngine.Initialize(std::string(outFolder));

	//Vector2i winSize;
	//int textHeight = 30; // Height of text area
	//winSize.x = 2 * MAX(appData->imageSource->getRGBImageSize().x, appData->imageSource->getDepthImageSize().x);
	//winSize.y = MAX(appData->imageSource->getRGBImageSize().y, appData->imageSource->getDepthImageSize().y) + textHeight;
	//float h1 = textHeight / (float)winSize.y, h2 = (1.f + h1) / 2;
	//winReg[0] = Vector4f(0, h1, 0.5, 1); // Main render
	//winReg[1] = Vector4f(0.5, h2, 0.75, 1); // Side sub window 0
	//winReg[2] = Vector4f(0.75, h2, 1, 1); // Side sub window 1
	//winReg[3] = Vector4f(0.5, h1, 0.75, h2); // Side sub window 2
	//winReg[4] = Vector4f(0.75, h1, 1, h2); // Side sub window 3

	int textHeight = 30; // Height of text area
	//winSize.x = (int)(1.5f * (float)MAX(appData->imageSource->getImageSize().x, appData->imageSource->getDepthImageSize().x));
	//winSize.y = MAX(appData->imageSource->getRGBImageSize().y, appData->imageSource->getDepthImageSize().y) + textHeight;
	winSize.x = (int)(1.5f * (float)(appData->imageSource->getDepthImageSize().x));
	winSize.y = appData->imageSource->getDepthImageSize().y + textHeight;
	float h1 = textHeight / (float)winSize.y, h2 = (1.f + h1) / 2;
	winReg[0] = Vector4f(0.0f, h1, 0.665f, 1.0f);   // Main render
	winReg[1] = Vector4f(0.665f, h2, 1.0f, 1.0f);   // Side sub window 0
	winReg[2] = Vector4f(0.665f, h1, 1.0f, h2);     // Side sub window 2

	this->isRecording = false;
	this->currentFrameNo = 0;
	this->rgbVideoWriter = nullptr;
	this->depthVideoWriter = nullptr;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(winSize.x, winSize.y);
	glutCreateWindow("InfiniTAM");
	glGenTextures(NUM_WIN, textureId);

	glutIgnoreKeyRepeat(true);
	glutDisplayFunc(UIEngine::glutDisplayFunction);
	glutReshapeFunc(UIEngine::glutReshape);
	glutKeyboardUpFunc(UIEngine::glutKeyUpFunction);
	glutKeyboardFunc(UIEngine::glutKeyFunction);
	glutMouseFunc(UIEngine::glutMouseButtonFunction);
	glutMotionFunc(UIEngine::glutMouseMoveFunction);
	glutIdleFunc(UIEngine::glutIdleFunction);

#ifdef FREEGLUT
	glutMouseWheelFunc(UIEngine::glutMouseWheelFunction);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, 1);
#endif

	bool allocateGPU = false;
	if (appData->internalSettings->deviceType == ITMLibSettings::DEVICE_CUDA) allocateGPU = true;

	for (int w = 0; w < NUM_WIN; w++)
		outImage[w] = new ITMUChar4Image(appData->imageSource->getDepthImageSize(), true, allocateGPU);

	inputRGBImage = new ITMUChar4Image(appData->imageSource->getRGBImageSize(), true, allocateGPU);
	inputRawDepthImage = new ITMShortImage(appData->imageSource->getDepthImageSize(), true, allocateGPU);
	inputIMUMeasurement = new ITMIMUMeasurement();

	saveImage = new ITMUChar4Image(appData->imageSource->getDepthImageSize(), true, false);

	outImageType[0] = ITMMainEngine::InfiniTAM_IMAGE_SCENERAYCAST;
	outImageType[1] = ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_DEPTH;
	outImageType[2] = ITMMainEngine::InfiniTAM_IMAGE_ORIGINAL_RGB;
	if (inputRGBImage->noDims == Vector2i(0, 0)) outImageType[2] = ITMMainEngine::InfiniTAM_IMAGE_UNKNOWN;
	//outImageType[3] = ITMMainEngine::InfiniTAM_IMAGE_SCENERAYCAST;
	//outImageType[4] = ITMMainEngine::InfiniTAM_IMAGE_SCENERAYCAST;

	mainLoopAction = PROCESS_PAUSED;
	mouseState = 0;
	mouseWarped = false;
	needsRefresh = false;
	processedFrameNo = 0;
	processedTime = 0.0f;

#ifndef COMPILE_WITHOUT_CUDA
	ORcudaSafeCall(cudaThreadSynchronize());
#endif

	sdkCreateTimer(&timer_instant);
	sdkCreateTimer(&timer_average);

	sdkResetTimer(&timer_average);

	printf("initialised.\n");
}

void UIEngine::SaveScreenshot(const char *filename) const
{
	ITMUChar4Image screenshot(getWindowSize(), true, false);
	GetScreenshot(&screenshot);
	SaveImageToFile(&screenshot, filename, true);
}

void UIEngine::GetScreenshot(ITMUChar4Image *dest) const
{
	glReadPixels(0, 0, dest->noDims.x, dest->noDims.y, GL_RGBA, GL_UNSIGNED_BYTE, dest->GetData(MEMORYDEVICE_CPU));
}

void SaveNormalImage(const ITMView *view, const std::string& path, ITMLibSettings::DeviceType deviceType)
{
	if (deviceType == ITMLibSettings::DEVICE_CUDA)
	{
		view->depthNormal->UpdateHostFromDevice();
	}

	ITMUChar4Image *normalImage = new ITMUChar4Image(view->rgb->noDims, true, false);
	Vector4u *data_to = normalImage->GetData(MEMORYDEVICE_CPU);
	const Vector4f *data_from = view->depthNormal->GetData(MEMORYDEVICE_CPU);

	for (int i=0; i < view->depthNormal->noDims[0] * view->depthNormal->noDims[1]; i++)
	{
		data_to[i].x = static_cast<uchar>(abs(data_from[i].x) * 255);
		data_to[i].y = static_cast<uchar>(abs(data_from[i].y) * 255);
		data_to[i].z = static_cast<uchar>(abs(data_from[i].z) * 255);
		data_to[i].w = 255;
	}
	SaveImageToFile(normalImage, path.c_str());
	free(normalImage);
}

void SaveNormalDirectionImage(const ITMView *view, const std::string& path, const Matrix4f& invM_d, ITMLibSettings::DeviceType deviceType)
{
	if (deviceType == ITMLibSettings::DEVICE_CUDA)
	{
		view->depthNormal->UpdateHostFromDevice();
	}

	ITMUChar4Image *normalImage = new ITMUChar4Image(view->rgb->noDims, true, false);
	Vector4u *data_to = normalImage->GetData(MEMORYDEVICE_CPU);
	const Vector4f *data_from = view->depthNormal->GetData(MEMORYDEVICE_CPU);

	for (int i=0; i < view->depthNormal->noDims[0] * view->depthNormal->noDims[1]; i++)
	{
		Vector4f normal_camera = data_from[i];
		normal_camera.w = 0;
		Vector4f normal_world = invM_d * normal_camera;

		float weights[N_DIRECTIONS];
		ComputeDirectionWeights(-normal_world.toVector3(), weights);

		const Vector3f directionColors[6] = {
			Vector3f(1, 0, 0),
			Vector3f(0, 1, 0),
			Vector3f(1, 1, 0),
			Vector3f(0, 0, 1),
			Vector3f(1, 0, 1),
			Vector3f(0, 1, 1)
		};

		float sumWeights = 0;
		Vector3f colorCombined(0, 0, 0);
		for (TSDFDirection_type direction = 0; direction < N_DIRECTIONS; direction++)
		{
			if (weights[direction] < direction_weight_threshold) continue;

			colorCombined += weights[direction] * directionColors[direction];
			sumWeights += weights[direction];
		}
		colorCombined /= sumWeights;
		data_to[i].x = static_cast<uchar>(abs(colorCombined.x) * 255);
		data_to[i].y = static_cast<uchar>(abs(colorCombined.y) * 255);
		data_to[i].z = static_cast<uchar>(abs(colorCombined.z) * 255);
		data_to[i].w = 255;

	}
	SaveImageToFile(normalImage, path.c_str());
	free(normalImage);
}

void SaveErrorImage(const ITMView *view, const std::string& path, ITMLibSettings::DeviceType deviceType)
{
	if (deviceType == ITMLibSettings::DEVICE_CUDA)
	{
		view->depthNormal->UpdateHostFromDevice();
	}

	ITMUChar4Image *errorImage = new ITMUChar4Image(view->rgb->noDims, true, false);
	SaveImageToFile(errorImage, path.c_str());
	free(errorImage);
}

void UIEngine::ProcessFrame()
{
	if (!appData->imageSource->hasMoreImages()) return;
	appData->imageSource->getImages(inputRGBImage, inputRawDepthImage);

	if (appData->imuSource != nullptr) {
		if (!appData->imuSource->hasMoreMeasurements()) return;
		else appData->imuSource->getMeasurement(inputIMUMeasurement);
	}

	const ORUtils::SE3Pose *inputPose = nullptr;
	if (appData->trajectorySource != nullptr)
	{
		if (!appData->trajectorySource->hasMorePoses()) return;
		inputPose = appData->trajectorySource->getPose();
	} else if (processedFrameNo == 0)
	{
		inputPose = &appData->initialPose;
	}

	sdkResetTimer(&timer_instant);
	sdkStartTimer(&timer_instant); sdkStartTimer(&timer_average);

	ITMTrackingState::TrackingResult trackerResult;
	//actual processing on the mailEngine
	if (appData->imuSource != nullptr) trackerResult = mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage, inputIMUMeasurement, inputPose);
	else trackerResult = mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage, nullptr, inputPose);

	trackingResult = (int)trackerResult;

	if (isRecording)
	{
		char str[250];

		sprintf(str, "%s/recording/%04d.pgm", outFolder, currentFrameNo);
		SaveImageToFile(inputRawDepthImage, str);

		if (inputRGBImage->noDims != Vector2i(0, 0)) {
			sprintf(str, "%s/recording/%04d.ppm", outFolder, currentFrameNo);
			SaveImageToFile(inputRGBImage, str);
		}

		sprintf(str, "%s/recording/render%04d.ppm", outFolder, currentFrameNo);
		SaveImageToFile(outImage[0], str);

		sprintf(str, "%s/recording/normal_%04d.ppm", outFolder, currentFrameNo);
		SaveNormalImage(mainEngine->GetView(), str, appData->internalSettings->deviceType);

		sprintf(str, "%s/recording/normal_direction_%04d.ppm", outFolder, currentFrameNo);
		SaveNormalDirectionImage(mainEngine->GetView(), str, mainEngine->GetTrackingState()->pose_d->GetInvM(), appData->internalSettings->deviceType);

		sprintf(str, "%s/recording/error_%04d.ppm", outFolder, currentFrameNo);
//		mainEngine->GetTrackingState()
//		mainEngine->GetView()
//		SaveErrorImage()
	}
	if ((rgbVideoWriter != nullptr) && (inputRGBImage->noDims.x != 0)) {
		if (!rgbVideoWriter->isOpen()) rgbVideoWriter->open("out_rgb.avi", inputRGBImage->noDims.x, inputRGBImage->noDims.y, false, 30);
		rgbVideoWriter->writeFrame(inputRGBImage);
	}
	if ((depthVideoWriter != nullptr) && (inputRawDepthImage->noDims.x != 0)) {
		if (!depthVideoWriter->isOpen()) depthVideoWriter->open("out_d.avi", inputRawDepthImage->noDims.x, inputRawDepthImage->noDims.y, true, 30);
		depthVideoWriter->writeFrame(inputRawDepthImage);
	}

#ifndef COMPILE_WITHOUT_CUDA
	ORcudaSafeCall(cudaThreadSynchronize());
#endif
	sdkStopTimer(&timer_instant); sdkStopTimer(&timer_average);

	// Safe input images
	inputImages.emplace_back();
//	inputImages.back().first = new ITMUChar4Image(true, false);
//	inputImages.back().first->SetFrom(inputRGBImage, ITMUChar4Image::CPU_TO_CPU);
	inputImages.back().first = inputRGBImage; // not required for error renderings, so only store reference
	inputImages.back().second = new ITMShortImage(true, false);
	inputImages.back().second->SetFrom(inputRawDepthImage, ITMShortImage::CPU_TO_CPU);
	trackingPoses.push_back(*mainEngine->GetTrackingState()->pose_d);

	statisticsEngine.LogTimeStats(mainEngine->GetTimeStats());
	statisticsEngine.LogPose(*mainEngine->GetTrackingState());
	statisticsEngine.LogBlockAllocations(mainEngine->GetAllocationsPerDirection());

	//processedTime = sdkGetTimerValue(&timer_instant);
	processedTime = sdkGetAverageTimerValue(&timer_average);

	currentFrameNo++;
}

void UIEngine::CollectICPErrorImages()
{
	if (inputImages.empty())
		return;

	ITMLib::ITMView* view = nullptr;
	ITMViewBuilder* viewBuilder = ITMViewBuilderFactory::MakeViewBuilder(appData->imageSource->getCalib(),
	                                                                     appData->internalSettings->deviceType);
	// needs to be called once with view=nullptr for initialization
	viewBuilder->UpdateView(&view, inputImages.front().first, inputImages.front().second,
	                        appData->internalSettings->useBilateralFilter);
	view = mainEngine->GetView();

	ITMUChar4Image* outputImage = new ITMUChar4Image(inputImages.front().first->noDims, true, false);
	char str[250];

	int lastPercentile = -1;
	for (size_t i = 0; i < inputImages.size(); i++)
	{
		int percentile = (i * 10) / inputImages.size();
		if (percentile != lastPercentile)
		{
			printf("%i%%\t", percentile * 10);
			lastPercentile = percentile;
		}
		ITMUChar4Image* rgbImage = inputImages.at(i).first;
		ITMShortImage* depthImage = inputImages.at(i).second;

		viewBuilder->UpdateView(&view, rgbImage, depthImage,
		                        appData->internalSettings->useBilateralFilter);

		ORUtils::SE3Pose* pose = &trackingPoses.at(i);
		mainEngine->GetTrackingState()->pose_d->SetFrom(pose);

//		renderState -> visible blocks??
		mainEngine->GetImage(outputImage, ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_ICP_ERROR,
		                     pose, &freeviewIntrinsics, normalsFromSDF);

		sprintf(str, "%s/recording/error_%04zu.ppm", outFolder, i);
		SaveImageToFile(outputImage, str);
	}

	free(outputImage);
	free(viewBuilder);
}

void UIEngine::Run() { glutMainLoop(); }
void UIEngine::Shutdown()
{
	sdkDeleteTimer(&timer_instant);
	sdkDeleteTimer(&timer_average);

	statisticsEngine.CloseAll();

	if (rgbVideoWriter != nullptr) delete rgbVideoWriter;
	if (depthVideoWriter != nullptr) delete depthVideoWriter;

	for (int w = 0; w < NUM_WIN; w++)
		delete outImage[w];

	delete inputRGBImage;
	delete inputRawDepthImage;
	delete inputIMUMeasurement;
	for (auto imgs: inputImages)
	{
//		delete imgs.first;
		delete imgs.second;
	}

	delete[] outFolder;
	delete saveImage;
	delete instance;
	instance = nullptr;
}
