# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/willy/Desktop/opencv/samples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/willy/Desktop/opencv/samples/dnn

# Include any dependencies generated for this target.
include dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/depend.make

# Include the progress variables for this target.
include dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/progress.make

# Include the compile flags for this target's objects.
include dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/flags.make

dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp.o: dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/flags.make
dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp.o: CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/willy/Desktop/opencv/samples/dnn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp.o"
	cd /home/willy/Desktop/opencv/samples/dnn/dnn && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example_dnn_OpenMPTryFlag.dir/CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp.o -c /home/willy/Desktop/opencv/samples/dnn/CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp

dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_dnn_OpenMPTryFlag.dir/CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp.i"
	cd /home/willy/Desktop/opencv/samples/dnn/dnn && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/willy/Desktop/opencv/samples/dnn/CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp > CMakeFiles/example_dnn_OpenMPTryFlag.dir/CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp.i

dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_dnn_OpenMPTryFlag.dir/CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp.s"
	cd /home/willy/Desktop/opencv/samples/dnn/dnn && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/willy/Desktop/opencv/samples/dnn/CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp -o CMakeFiles/example_dnn_OpenMPTryFlag.dir/CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp.s

# Object files for target example_dnn_OpenMPTryFlag
example_dnn_OpenMPTryFlag_OBJECTS = \
"CMakeFiles/example_dnn_OpenMPTryFlag.dir/CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp.o"

# External object files for target example_dnn_OpenMPTryFlag
example_dnn_OpenMPTryFlag_EXTERNAL_OBJECTS =

dnn/example_dnn_OpenMPTryFlag: dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/CMakeFiles/FindOpenMP/OpenMPTryFlag.cpp.o
dnn/example_dnn_OpenMPTryFlag: dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/build.make
dnn/example_dnn_OpenMPTryFlag: /usr/local/lib/libopencv_objdetect.so.4.5.5
dnn/example_dnn_OpenMPTryFlag: /usr/local/lib/libopencv_video.so.4.5.5
dnn/example_dnn_OpenMPTryFlag: /usr/local/lib/libopencv_highgui.so.4.5.5
dnn/example_dnn_OpenMPTryFlag: /usr/local/lib/libopencv_dnn.so.4.5.5
dnn/example_dnn_OpenMPTryFlag: /usr/local/lib/libopencv_calib3d.so.4.5.5
dnn/example_dnn_OpenMPTryFlag: /usr/local/lib/libopencv_features2d.so.4.5.5
dnn/example_dnn_OpenMPTryFlag: /usr/local/lib/libopencv_flann.so.4.5.5
dnn/example_dnn_OpenMPTryFlag: /usr/local/lib/libopencv_videoio.so.4.5.5
dnn/example_dnn_OpenMPTryFlag: /usr/local/lib/libopencv_imgcodecs.so.4.5.5
dnn/example_dnn_OpenMPTryFlag: /usr/local/lib/libopencv_imgproc.so.4.5.5
dnn/example_dnn_OpenMPTryFlag: /usr/local/lib/libopencv_core.so.4.5.5
dnn/example_dnn_OpenMPTryFlag: dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/willy/Desktop/opencv/samples/dnn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example_dnn_OpenMPTryFlag"
	cd /home/willy/Desktop/opencv/samples/dnn/dnn && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_dnn_OpenMPTryFlag.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/build: dnn/example_dnn_OpenMPTryFlag

.PHONY : dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/build

dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/clean:
	cd /home/willy/Desktop/opencv/samples/dnn/dnn && $(CMAKE_COMMAND) -P CMakeFiles/example_dnn_OpenMPTryFlag.dir/cmake_clean.cmake
.PHONY : dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/clean

dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/depend:
	cd /home/willy/Desktop/opencv/samples/dnn && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/willy/Desktop/opencv/samples /home/willy/Desktop/opencv/samples/dnn /home/willy/Desktop/opencv/samples/dnn /home/willy/Desktop/opencv/samples/dnn/dnn /home/willy/Desktop/opencv/samples/dnn/dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dnn/CMakeFiles/example_dnn_OpenMPTryFlag.dir/depend

