// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "UIEngine.h"

#include <string.h>
#include <GL/glut.h>

#ifdef FREEGLUT
#include <GL/freeglut.h>
#include <ITMLib/Engines/ITMViewBuilderFactory.h>
#include <ITMLib/Utils/ITMProjectionUtils.h>
#endif

#include <ITMLib/ITMLibDefines.h>
#include <ITMLib/Core/ITMBasicEngine.h>

#include <ORUtils/FileUtils.h>
#include <InputSource/FFMPEGWriter.h>

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
	uiEngine->mainEngine->GetImage(uiEngine->outImage[0], uiEngine->outImageType[0], &uiEngine->freeviewPose, &uiEngine->freeviewIntrinsics, uiEngine->appData->internalSettings->useSDFNormals);

	for (int w = 1; w < NUM_WIN; w++) uiEngine->mainEngine->GetImage(uiEngine->outImage[w], uiEngine->outImageType[w], nullptr, nullptr, uiEngine->appData->internalSettings->useSDFNormals);

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
		        uiEngine->appData->internalSettings->useSDFNormals ? "from SDF" : "from points",
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
		uiEngine->ProcessFrame();
		uiEngine->mainLoopAction = PROCESS_PAUSED;
		uiEngine->needsRefresh = true;
		break;
	case PROCESS_VIDEO:
		uiEngine->ProcessFrame();
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
			uiEngine->isRecording = true;
		}
		break;
	case 'v':
		if ((uiEngine->rgbVideoWriter != nullptr) || (uiEngine->depthVideoWriter != nullptr) || (uiEngine->outputVideoWriter != nullptr))
		{
			printf("stop recoding video\n");
			delete uiEngine->rgbVideoWriter;
			delete uiEngine->depthVideoWriter;
			delete uiEngine->outputVideoWriter;
			uiEngine->rgbVideoWriter = nullptr;
			uiEngine->depthVideoWriter = nullptr;
			uiEngine->outputVideoWriter = nullptr;
		}
		else
		{
			printf("start recoding video\n");
			uiEngine->rgbVideoWriter = new FFMPEGWriter();
			uiEngine->depthVideoWriter = new FFMPEGWriter();
			uiEngine->outputVideoWriter = new FFMPEGWriter();
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

			uiEngine->freeviewActive = true;
		}
		uiEngine->needsRefresh = true;
		break;
	case '1': case '2': case '3': case '4': case '5': case '6':
		uiEngine->currentColourMode = (key - '1') % (uiEngine->freeviewActive ? 5 : 6);
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
		uiEngine->appData->internalSettings->useSDFNormals = !uiEngine->appData->internalSettings->useSDFNormals;
		uiEngine->needsRefresh = true;
		break;
	case 't':
	{
		uiEngine->integrationActive = !uiEngine->integrationActive;

		ITMBasicEngine *basicEngine = dynamic_cast<ITMBasicEngine*>(uiEngine->mainEngine);
		if (basicEngine != nullptr)
		{
			if (uiEngine->integrationActive) basicEngine->turnOnIntegration();
			else basicEngine->turnOffIntegration();
		}
	}
	break;
	case 'e':
		printf("Collecting ICP Error Images ... ");
		uiEngine->CollectICPErrorImages();
//		uiEngine->CollectPointClouds();
		printf("done\n");
		break;
	case 'm':
	{
		printf("saving scene to model ... ");
		uiEngine->mainEngine->SaveSceneToMesh("mesh.ply");
		printf("done\n");
	}
	break;
	case 'r':
	{
		ITMBasicEngine *basicEngine = dynamic_cast<ITMBasicEngine*>(uiEngine->mainEngine);
		if (basicEngine != nullptr) basicEngine->resetAll();
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

void UIEngine::printPixelInformation(int x, int y)
{
	Vector2f c1(winReg[0][0], winReg[0][1]);
	Vector2f c2(winReg[0][2], winReg[0][1]);
	Vector2f c3(winReg[0][2], winReg[0][3]);
	Vector2f c4(winReg[0][0], winReg[0][3]);
	Vector2f p_window(static_cast<float>(x) / glutGet(GLUT_WINDOW_WIDTH), static_cast<float>(y) / glutGet(GLUT_WINDOW_HEIGHT));

	Vector2f p_image(
		(p_window.x - c1.x) / (c2.x - c1.x),
		1 - (c4.y - p_window.y - c1.y) / (c4.y - c1.y)
		);

	Vector2i imgSize;
	if (freeviewActive)
		imgSize = freeviewIntrinsics.imgSize;
	else
		imgSize = mainEngine->GetImageSize();
	Vector2i idx_image(imgSize.width * p_image.x, imgSize.height * p_image.y);
	if (idx_image.x < 0 or idx_image.x >= imgSize.width or idx_image.y < 0 or idx_image.y >= imgSize.height)
		return;

	Vector4f point;
	if (freeviewActive)
	{
		if (appData->internalSettings->deviceType == ITMLibSettings::DEVICE_CUDA)
		{
			ORUtils::Image<Vector4f> image(mainEngine->GetRenderStateFreeview()->raycastResult->noDims, MEMORYDEVICE_CPU);
			image.SetFrom(mainEngine->GetRenderStateFreeview()->raycastResult, ORUtils::CUDA_TO_CPU);

			point = image.GetData(MEMORYDEVICE_CPU)[imgSize.width * idx_image.y + idx_image.x];
		}
		else
			point = mainEngine->GetRenderStateFreeview()->raycastResult->GetData(MEMORYDEVICE_CPU)[imgSize.width * idx_image.y + idx_image.x];
	}
	else
	{
		if (appData->internalSettings->deviceType == ITMLibSettings::DEVICE_CUDA)
		{
			ORUtils::Image<Vector4f> image(mainEngine->GetRenderState()->raycastResult->noDims, MEMORYDEVICE_CPU);
			image.SetFrom(mainEngine->GetRenderState()->raycastResult, ORUtils::CUDA_TO_CPU);
			point = image.GetData(MEMORYDEVICE_CPU)[imgSize.width * idx_image.y + idx_image.x];
		}
		else
			point = mainEngine->GetRenderState()->raycastResult->GetData(MEMORYDEVICE_CPU)[imgSize.width * idx_image.y + idx_image.x];
	}

	if (point.w <= 0)
	{
		printf("img(xy)[%i, %i]\tNo data\n", idx_image.x, idx_image.y);
		return;
	}

	const float voxelSize = this->appData->internalSettings->sceneParams.voxelSize;
	Vector3i voxelIdx = (1 / voxelSize * point.toVector3()).toInt();
	Vector3i blockIdx;
	unsigned short offset;
	voxelToBlockPosAndOffset(voxelIdx, blockIdx, offset);

	printf("img(xy)[%i, %i]\tpoint(xyz)(%f, %f, %f)\tvoxel(xyz)(%i, %i, %i)	block(xyz)(%i, %i, %i)\n",
		idx_image.x, idx_image.y, point.x, point.y, point.z,
		voxelIdx.x, voxelIdx.y, voxelIdx.z, blockIdx.x, blockIdx.y, blockIdx.z
		);
}

void UIEngine::glutMouseButtonFunction(int button, int state, int x, int y)
{
	UIEngine *uiEngine = UIEngine::Instance();

	if (state == GLUT_DOWN)
	{
		switch (button)
		{
		case GLUT_LEFT_BUTTON: uiEngine->mouseState = 1;
			uiEngine->printPixelInformation(x, y);
			break;
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

void UIEngine::_initialise(int argc, char** argv, AppData* appData, ITMMainEngine *mainEngine)
{
	this->freeviewActive = false;
	this->integrationActive = true;
	this->helpActive = false;
	this->renderAxesActive = true;
	this->currentColourMode = 0;
	memset(keysPressed, false, sizeof(keysPressed));
	this->colourModes_main.push_back(UIColourMode("shaded greyscale", ITMMainEngine::InfiniTAM_IMAGE_SCENERAYCAST));
	this->colourModes_main.push_back(UIColourMode("integrated colours", ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_VOLUME));
	this->colourModes_main.push_back(UIColourMode("surface normals", ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_NORMAL));
	this->colourModes_main.push_back(UIColourMode("confidence", ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_CONFIDENCE));
	this->colourModes_main.push_back(UIColourMode("depth", ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_DEPTH));
	this->colourModes_main.push_back(UIColourMode("icp error", ITMMainEngine::InfiniTAM_IMAGE_COLOUR_FROM_ICP_ERROR));
	this->colourModes_freeview.push_back(UIColourMode("shaded greyscale", ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_SHADED));
	this->colourModes_freeview.push_back(UIColourMode("integrated colours", ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME));
	this->colourModes_freeview.push_back(UIColourMode("surface normals", ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL));
	this->colourModes_freeview.push_back(UIColourMode("confidence", ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_CONFIDENCE));
	this->colourModes_freeview.push_back(UIColourMode("depth", ITMMainEngine::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_DEPTH));

	int textHeight = 30; // Height of text area
	winSize.x = (int)(1.5f * (float)(appData->imageSource->getDepthImageSize().x));
	winSize.y = appData->imageSource->getDepthImageSize().y + textHeight;
	float h1 = textHeight / (float)winSize.y, h2 = (1.f + h1) / 2;
	winReg[0] = Vector4f(0.0f, h1, 0.665f, 1.0f);   // Main render
	winReg[1] = Vector4f(0.665f, h2, 1.0f, 1.0f);   // Side sub window 0
	winReg[2] = Vector4f(0.665f, h1, 1.0f, h2);     // Side sub window 2

	this->isRecording = false;
	this->rgbVideoWriter = nullptr;
	this->depthVideoWriter = nullptr;
	this->outputVideoWriter = nullptr;

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
	processedTime = 0.0f;
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

	ITMUChar4Image *normalImage = new ITMUChar4Image(view->depthNormal->noDims, true, false);
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
		ComputeDirectionAngle(-normal_world.toVector3(), weights);

		float sumWeights = 0;
		Vector3f colorCombined(0, 0, 0);
		for (TSDFDirection_type direction = 0; direction < N_DIRECTIONS; direction++)
		{
			float weight = DirectionWeight(weights[direction]);

			colorCombined += weight * TSDFDirectionColor[direction];
			sumWeights += weight;
		}
		if (sumWeights > 0)
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

bool UIEngine::_processFrame()
{
	glutTimerFunc(5000, UIEngine::checkStuck, currentFrameNo);

	ITMTrackingState::TrackingResult trackerResult;
	//actual processing on the mailEngine
	if (appData->imuSource != nullptr) trackerResult = mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage, inputIMUMeasurement, inputPose);
	else trackerResult = mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage, nullptr, inputPose);

	trackingResult = (int)trackerResult;

	if (isRecording)
	{
		fs::path recordingFolder = fs::path(outFolder) / "recording";
		fs::create_directories(recordingFolder);
		char str[250];

//		sprintf(str, "%s/recording/depth%04d.pgm", outFolder, currentFrameNo);
//		SaveImageToFile(inputRawDepthImage, str);

//		{
//			mainEngine->GetView()->depth->UpdateHostFromDevice();
//			ORUtils::Image<ORUtils::Vector4<float>>* points = new ORUtils::Image<ORUtils::Vector4<float>>(
//				mainEngine->GetView()->calib.intrinsics_d.imgSize, true, false);
//			Matrix4f invM = mainEngine->GetTrackingState()->pose_d->GetInvM();
//
//			Vector4f invProjParams = invertProjectionParams(mainEngine->GetView()->calib.intrinsics_d.projectionParamsSimple.all);
//			for (int y = 0; y < mainEngine->GetImageSize().height; y++)
//					for (int x = 0; x < mainEngine->GetImageSize().width; x++)
//			{
//				int i = y * mainEngine->GetImageSize().width + x;
//				float depth = mainEngine->GetView()->depth->GetData(MEMORYDEVICE_CPU)[i];
//				if (depth <= 0)
//				{
//					points->GetData(MEMORYDEVICE_CPU)[i] = Vector4f(0, 0, 0, 0);
//					continue;
//				}
//				Vector3f pt_camera = reprojectImagePoint(x, y, depth, invProjParams);
//				points->GetData(MEMORYDEVICE_CPU)[i] = Vector4f(pt_camera, 1);
//			}
//
//			mainEngine->GetView()->depth->GetData(MEMORYDEVICE_CPU),
//
//				sprintf(str, "%s/recording/depth_cloud_%04d.pcd", outFolder, currentFrameNo);
//
//			mainEngine->GetView()->depthNormal->UpdateHostFromDevice();
//
//			SavePointCloudToPCL(
//				points->GetData(MEMORYDEVICE_CPU),
//				mainEngine->GetView()->depthNormal->GetData(MEMORYDEVICE_CPU),
//				inputImages.front().second->noDims, *(mainEngine->GetTrackingState()->pose_d), str);
//			free(points);
//		}

		if (inputRGBImage->noDims != Vector2i(0, 0)) {
			sprintf(str, "%s/recording/rgb%04d.ppm", outFolder, currentFrameNo);
			SaveImageToFile(inputRGBImage, str);
		}

		sprintf(str, "%s/recording/render%04d.ppm", outFolder, currentFrameNo);
		SaveImageToFile(outImage[0], str);
//
//		sprintf(str, "%s/recording/normal%04d.ppm", outFolder, currentFrameNo);
//		SaveNormalImage(mainEngine->GetView(), str, appData->internalSettings->deviceType);
//
//		sprintf(str, "%s/recording/normal_direction%04d.ppm", outFolder, currentFrameNo);
//		SaveNormalDirectionImage(mainEngine->GetView(), str, mainEngine->GetTrackingState()->pose_d->GetInvM(), appData->internalSettings->deviceType);
//
//		sprintf(str, "%s/recording/depth_color%04d.pgm", outFolder, currentFrameNo);
//		mainEngine->GetImage(outImage[0], ITMMainEngine::GetImageType::InfiniTAM_IMAGE_ORIGINAL_DEPTH, &freeviewPose, &freeviewIntrinsics, uiEngine->appData->internalSettings->useSDFNormals);
//		SaveImageToFile(outImage[0], str);

//		sprintf(str, "%s/recording/error%04d.ppm", outFolder, currentFrameNo);
//		mainEngine->GetTrackingState()
//		mainEngine->GetView()
//		SaveErrorImage()
	}

	// Write input RGB to video
	if ((rgbVideoWriter != nullptr) && (inputRGBImage->noDims.x != 0)) {
		char str[250];
		sprintf(str, "%s/recording/in_rgb.avi", outFolder);
		if (!rgbVideoWriter->isOpen()) rgbVideoWriter->open(str, inputRGBImage->noDims.x, inputRGBImage->noDims.y, false, 30);
		rgbVideoWriter->writeFrame(inputRGBImage);
	}
	// Write input depth to video
	if ((depthVideoWriter != nullptr) && (inputRawDepthImage->noDims.x != 0)) {
		char str[250];
		sprintf(str, "%s/recording/in_d.avi", outFolder);
		if (!depthVideoWriter->isOpen()) depthVideoWriter->open(str, inputRawDepthImage->noDims.x, inputRawDepthImage->noDims.y, true, 30);
		depthVideoWriter->writeFrame(inputRawDepthImage);
	}

	// Write output rendering to video
	if ((outputVideoWriter != nullptr) && (inputRawDepthImage->noDims.x != 0)) {
		char str[250];
		sprintf(str, "%s/recording/out.avi", outFolder);
		if (!outputVideoWriter->isOpen()) outputVideoWriter->open(str, inputRawDepthImage->noDims.x, inputRawDepthImage->noDims.y, false, 30);
		outputVideoWriter->writeFrame(outImage[0]);
	}

#ifndef COMPILE_WITHOUT_CUDA
	ORcudaSafeCall(cudaThreadSynchronize());
#endif


	//processedTime = sdkGetTimerValue(&timer_instant);
	processedTime = sdkGetAverageTimerValue(&timer_average);

	return true;
}

void UIEngine::Run() { glutMainLoop(); }
void UIEngine::Shutdown()
{
	statisticsEngine.CloseAll();

	if (rgbVideoWriter != nullptr) delete rgbVideoWriter;
	if (depthVideoWriter != nullptr) delete depthVideoWriter;
	if (outputVideoWriter != nullptr) delete outputVideoWriter;

	for (int w = 0; w < NUM_WIN; w++)
		delete outImage[w];

	delete[] outFolder;
	delete saveImage;
	delete instance;
	instance = nullptr;
}

void UIEngine::checkStuck(int frameNoBefore)
{
	UIEngine *uiEngine = UIEngine::Instance();

	if (uiEngine->currentFrameNo <= frameNoBefore)
	{
		printf("Stuck while processing frame. Exiting program.");
		exit(-2);
	}
}
