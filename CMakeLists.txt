cmake_minimum_required(VERSION 3.0.0)
project(MouseTracker_KF VERSION 0.1.0)

# INCLUDE_DIRECTORIES(
#     "C:/GIT/vcpkg/installed/x64-windows/include/opencv2",
#     "C:/GIT/vcpkg/installed/x64-windows/include/", 
#     "C:/GIT/vcpkg/installed/x64-windows/include/opencv2/video"

# )

# LINK_DIRECTORIES(
#     C:/GIT/vcpkg/installed/x64-windows/lib
# )
# add_executable(MouseTracker_KF main2.cpp)

add_executable(MouseTracker_KF TrackMousePointer.cpp)


# TARGET_LINK_LIBRARIES(MouseTracker_KF
#     opencv2
# )


