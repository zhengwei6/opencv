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
CMAKE_SOURCE_DIR = /home/willy/Desktop/opencv/samples/mobile_net

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/willy/Desktop/opencv/samples/mobile_net

# Include any dependencies generated for this target.
include CMakeFiles/Classification.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Classification.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Classification.dir/flags.make

CMakeFiles/Classification.dir/classification.cpp.o: CMakeFiles/Classification.dir/flags.make
CMakeFiles/Classification.dir/classification.cpp.o: classification.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/willy/Desktop/opencv/samples/mobile_net/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Classification.dir/classification.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Classification.dir/classification.cpp.o -c /home/willy/Desktop/opencv/samples/mobile_net/classification.cpp

CMakeFiles/Classification.dir/classification.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Classification.dir/classification.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/willy/Desktop/opencv/samples/mobile_net/classification.cpp > CMakeFiles/Classification.dir/classification.cpp.i

CMakeFiles/Classification.dir/classification.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Classification.dir/classification.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/willy/Desktop/opencv/samples/mobile_net/classification.cpp -o CMakeFiles/Classification.dir/classification.cpp.s

# Object files for target Classification
Classification_OBJECTS = \
"CMakeFiles/Classification.dir/classification.cpp.o"

# External object files for target Classification
Classification_EXTERNAL_OBJECTS =

Classification: CMakeFiles/Classification.dir/classification.cpp.o
Classification: CMakeFiles/Classification.dir/build.make
Classification: /usr/local/lib/libopencv_gapi.so.4.5.5
Classification: /usr/local/lib/libopencv_highgui.so.4.5.5
Classification: /usr/local/lib/libopencv_ml.so.4.5.5
Classification: /usr/local/lib/libopencv_objdetect.so.4.5.5
Classification: /usr/local/lib/libopencv_photo.so.4.5.5
Classification: /usr/local/lib/libopencv_stitching.so.4.5.5
Classification: /usr/local/lib/libopencv_video.so.4.5.5
Classification: /usr/local/lib/libopencv_videoio.so.4.5.5
Classification: /usr/local/lib/libopencv_imgcodecs.so.4.5.5
Classification: /usr/local/lib/libopencv_dnn.so.4.5.5
Classification: /usr/local/lib/libopencv_calib3d.so.4.5.5
Classification: /usr/local/lib/libopencv_features2d.so.4.5.5
Classification: /usr/local/lib/libopencv_flann.so.4.5.5
Classification: /usr/local/lib/libopencv_imgproc.so.4.5.5
Classification: /usr/local/lib/libopencv_core.so.4.5.5
Classification: CMakeFiles/Classification.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/willy/Desktop/opencv/samples/mobile_net/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Classification"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Classification.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Classification.dir/build: Classification

.PHONY : CMakeFiles/Classification.dir/build

CMakeFiles/Classification.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Classification.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Classification.dir/clean

CMakeFiles/Classification.dir/depend:
	cd /home/willy/Desktop/opencv/samples/mobile_net && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/willy/Desktop/opencv/samples/mobile_net /home/willy/Desktop/opencv/samples/mobile_net /home/willy/Desktop/opencv/samples/mobile_net /home/willy/Desktop/opencv/samples/mobile_net /home/willy/Desktop/opencv/samples/mobile_net/CMakeFiles/Classification.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Classification.dir/depend

