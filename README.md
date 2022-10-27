# Directional TSDF InfiniTAM

This repository contains an implementation of Directional TSDF (DTSDF) and is based on InfiniTAM v3. If you use our code
for your publications, please cite our work (see [Research](#3-research))

Previous maintainers and contributors are:

Malte Splietker  
Victor Adrian Prisacariu <victor@robots.ox.ac.uk>  
Olaf Kaehler <olaf@robots.ox.ac.uk>  
Stuart Golodetz <smg@robots.ox.ac.uk>  
Michael Sapienza <michael.sapienza@eng.ox.ac.uk>  
Tommaso Cavallari <tommaso.cavallari@unibo.it>  
Carl Yuheng Ren <carl@robots.ox.ac.uk>  
Ming Ming Cheng <cmm.thu@gmail.com>  
Xin Sun <xin.sun@st-hughs.ox.ac.uk>  
Philip H.S. Torr <philip.torr@eng.ox.ac.uk>  
Ian D Reid <ian.reid@adelaide.edu.au>  
David W Murray <dwm@robots.ox.ac.uk>

## 1. Building the System

The system can be build in different ways. For convenience we also added a __conda__ environment, as well as a __Docker__ container.

Tested on a ubuntu 18.04 and 20.04 host systems with CUDA versions 10.1 and 11.4.

### 1.1 Anaconda ###
Probably the easiest way is to set up a conda environment, which contains all required libraries.
```bash
cd path/to/DirectionalTSDF
conda env create -f environment.yml
conda activate dtsfd
./build.sh
build/Apps/InfiniTAM/InfiniTAM --other-options --see-below ...
```
This method still requires `libgl` and `libglx` to be installed on the host system.

### 1.2 Docker ###
The directory `docker` contains a Dockerfile and scripts for building and running the container.

The container is set up as a development environment, i.e., the current source directory is mounted inside the container so any alterations are available on both host- and docker system. The username and user-id are copied to the container, so file modifications don't cause permission problems.

To build the container simply call
```bash
docker/build.sh
```
Then run the container by calling
```bash
docker/run.sh
```
and build and run the code with
```bash
./build.sh
build/Apps/InfiniTAM/InfiniTAM --other-options --see-below ...
```

You can modify the `run.sh` to mount any additionally required directories (e.g. datasets), see commented line inside the file.

### 1.3 Manual Installation ###
### 1.3.1 Requirements

Several 3rd party libraries are needed for compiling InfiniTAM. The given version numbers are checked and working, but
different versions might be fine as well. Some of the libraries are optional, and skipping them will reduce
functionality.

#### Required

- cmake >= 3.13
- libopengl
- libglu
- libglx
- freeglut
- CUDA >= 9

```bash
sudo apt install git cmake libeigen3-dev freeglut3-dev zlib1g-dev libpng-dev libyaml-cpp-dev
```

#### Optional

- Clang (>=8)
  Builds faster than GCC
  
- OpenNI (e.g. version 2.2.0.33)
  Allows get live images from OpenNI hardware. Also make sure you have freenect/OpenNI2-FreenectDriver if you need it available at http://structure.io/openni

- libpng (e.g. version 1.6)
  Allows to read and write PNG files available at http://libpng.org

- FFMPEG (e.g. version 2.8.6)
  Allows writing and playback of lossless FFV1 encoded videos available at https://www.ffmpeg.org/

- librealsense (e.g. github version from 2016-MAR-22)
  Allows to get live images from Intel Realsense cameras available at https://github.com/IntelRealSense/librealsense

- librealsense2 (e.g. Intel® RealSense™ SDK 2.X)
  Allows to get live images from Intel Realsense cameras available at https://github.com/IntelRealSense/librealsense

- doxygen (e.g. version 1.8.2)
  Builds a nice reference manual available at http://www.doxygen.org/

### 1.3.2 Build Process

To compile the system, use the standard cmake approach (use options for required input devices, e.g. by using ccmake). For example
```
git submodule update --init --recursive
mkdir build
cd build
cmake -DWITH_PNG=ON -DWITH_OPENNI=ON -DWITH_REALSENSE2=ON -DREALSENSE2_ROOT="/usr/" -DOPENNI_ROOT="/usr/" -DWITH_KINECT2=ON ..
make
```
alternatively use the provided `build.sh` script.

To create a doxygen documentation, just run doxygen:

```
doxygen Doxyfile
```

This will create a new directory doxygen-html/ containing all the documentation.

## 2. Sample Programs

The build process should result in two executables, ```InfiniTAM``` and ```InfiniTAM_cli```. The former launches a GUI, the latter is a headless command line application. If compiled with for example OpenNI support, both should run out-of-the-box without problems for live reconstruction. All available command line options are printed using the ```--help``` flag. If no device support has been compiled in, the program can be used for offline processing. For raw datasets in the form of

```
build/Apps/InfiniTAM/InfiniTAM --calibration Teddy/calib.txt --settings ./Files/settings.yaml --raw Teddy/Frames/%04i.ppm Teddy/Frames/%04i.pgm
```
The arguments are essentially masks for sprintf and the %04i will be replaced by a running number, accordingly. Supported formats are `ppm` and `png` for color images and `pgm` and `png` for depth images.

Datasets in [TUM](https://vision.in.tum.de/data/datasets/rgbd-dataset) format are also supported. Here is an example for a dataset from the fr3 sequences. (Note: the dataset's
rgb.txt and depth.txt must only contain matched pairs of rgb and depth images.

```
build/Apps/InfiniTAM/InfiniTAM --calibration ./Files/TUM3.txt --settings ./Files/settings.yaml --tum /path/to/dataset
```

The calibration files (e.g. ```.Files/TUM3.txt```) contain camera calibrations specific for each datasets. Many live
input sources like OpenNI2 automatically provide their intrinsics via the respective library. The file ```./Files/settings.yaml``` contains algorithm parameters like voxel size, allocation sizes tracking parameters etc.

The GUI application shows a help screen by pressing `h`.

Statistics and other output are written to the ```Output``` directory of the present working directory, if it exists.

# 3. Research
Original paper (IROS 2019) introducing the Directional TSDF and modified Marching Cubes
algorithm. [PDF](http://ais.uni-bonn.de/papers/IROS_2019_Splietker.pdf)

```
@InProceedings{DTSDF_IROS_2019,
  author    = {M. {Splietker} and S. {Behnke}},
  title     = {Directional {TSDF}: Modeling Surface Orientation for Coherent Meshes},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2019},
  pages     = {1727--1734}
}
```

Most recent paper (ECMR 2021) including ray-casting rendering, combined TSDF, ICP tracking and color
fusion. [PDF](https://arxiv.org/abs/2108.08115)

```
@misc{DTSDF_ECMR_2021,,
      title={Rendering and Tracking the Directional {TSDF}: Modeling Surface Orientation for Coherent Maps}, 
      author={Malte {Splietker} and Sven {Behnke}},
      year={2021},
      eprint={2108.08115},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

For more information about the original InfiniTAM please visit the project website <http://www.infinitam.org>.