cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST)

include_directories(BEFORE ..)

set(EXECUTABLE ethtester)

file(GLOB HEADERS "*.h")

add_executable(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

add_dependencies(${EXECUTABLE} BuildInfo.h)

target_link_libraries(${EXECUTABLE} ethcore)
target_link_libraries(${EXECUTABLE} ethash)
target_link_libraries(${EXECUTABLE} devcore libjson-rpc-cpp::client)

if(ETHDBUS)
	find_package(PkgConfig)
	set( ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/usr/lib/x86_64-linux-gnu/pkgconfig" )
	pkg_check_modules(DBUS dbus-1)
	include_directories(${DBUS_INCLUDE_DIRS})
	link_directories(${DBUS_LIBRARY_DIRS})
	target_link_libraries(${EXECUTABLE} ${DBUS_LIBRARIES})
endif()

if(APICORE)
	target_link_libraries(${EXECUTABLE} apicore)
endif()

if(ETHSTRATUM)
    target_link_libraries(${EXECUTABLE} ethstratum)
endif()

include(GNUInstallDirs)
