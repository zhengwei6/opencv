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
include cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/depend.make

# Include the progress variables for this target.
include cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/progress.make

# Include the compile flags for this target's objects.
include cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/flags.make

cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/intelligent_scissors.cpp.o: cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/flags.make
cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/intelligent_scissors.cpp.o: ../cpp/intelligent_scissors.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/willy/Desktop/opencv/samples/dnn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/intelligent_scissors.cpp.o"
	cd /home/willy/Desktop/opencv/samples/dnn/cpp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example_cpp_intelligent_scissors.dir/intelligent_scissors.cpp.o -c /home/willy/Desktop/opencv/samples/cpp/intelligent_scissors.cpp

cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/intelligent_scissors.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_cpp_intelligent_scissors.dir/intelligent_scissors.cpp.i"
	cd /home/willy/Desktop/opencv/samples/dnn/cpp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/willy/Desktop/opencv/samples/cpp/intelligent_scissors.cpp > CMakeFiles/example_cpp_intelligent_scissors.dir/intelligent_scissors.cpp.i

cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/intelligent_scissors.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_cpp_intelligent_scissors.dir/intelligent_scissors.cpp.s"
	cd /home/willy/Desktop/opencv/samples/dnn/cpp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/willy/Desktop/opencv/samples/cpp/intelligent_scissors.cpp -o CMakeFiles/example_cpp_intelligent_scissors.dir/intelligent_scissors.cpp.s

# Object files for target example_cpp_intelligent_scissors
example_cpp_intelligent_scissors_OBJECTS = \
"CMakeFiles/example_cpp_intelligent_scissors.dir/intelligent_scissors.cpp.o"

# External object files for target example_cpp_intelligent_scissors
example_cpp_intelligent_scissors_EXTERNAL_OBJECTS =

cpp/example_cpp_intelligent_scissors: cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/intelligent_scissors.cpp.o
cpp/example_cpp_intelligent_scissors: cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/build.make
cpp/example_cpp_intelligent_scissors: /usr/local/lib/libopencv_highgui.so.4.5.5
cpp/example_cpp_intelligent_scissors: /usr/local/lib/libopencv_videoio.so.4.5.5
cpp/example_cpp_intelligent_scissors: /usr/local/lib/libopencv_imgcodecs.so.4.5.5
cpp/example_cpp_intelligent_scissors: /usr/local/lib/libopencv_imgproc.so.4.5.5
cpp/example_cpp_intelligent_scissors: /usr/local/lib/libopencv_core.so.4.5.5
cpp/example_cpp_intelligent_scissors: cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/willy/Desktop/opencv/samples/dnn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example_cpp_intelligent_scissors"
	cd /home/willy/Desktop/opencv/samples/dnn/cpp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_cpp_intelligent_scissors.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/build: cpp/example_cpp_intelligent_scissors

.PHONY : cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/build

cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/clean:
	cd /home/willy/Desktop/opencv/samples/dnn/cpp && $(CMAKE_COMMAND) -P CMakeFiles/example_cpp_intelligent_scissors.dir/cmake_clean.cmake
.PHONY : cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/clean

cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/depend:
	cd /home/willy/Desktop/opencv/samples/dnn && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/willy/Desktop/opencv/samples /home/willy/Desktop/opencv/samples/cpp /home/willy/Desktop/opencv/samples/dnn /home/willy/Desktop/opencv/samples/dnn/cpp /home/willy/Desktop/opencv/samples/dnn/cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cpp/CMakeFiles/example_cpp_intelligent_scissors.dir/depend

