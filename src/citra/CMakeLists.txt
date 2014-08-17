set(SRCS    citra.cpp
            emu_window/emu_window_glfw.cpp)
set(HEADERS citra.h
            resource.h)

# NOTE: This is a workaround for CMake bug 0006976 (missing X11_xf86vmode_LIB variable)
if (NOT X11_xf86vmode_LIB)
    set(X11_xv86vmode_LIB Xxf86vm)
endif()

add_executable(citra ${SRCS} ${HEADERS})

if (NOT APPLE)
    set(EXTRA_X11_LIBS Xi Xcursor ${X11_Xrandr_LIB} ${X11_xv86vmode_LIB})
endif()

target_link_libraries(citra core common video_core ${ICONV_LIBRARY} pthread  ${COREFOUNDATION_LIBRARY} ${OPENGL_LIBRARIES} ${EXTRA_X11_LIBS} ${GLEW_LIBRARY} ${GLFW_LIBRARIES} ${RT_LIBRARY})

#install(TARGETS citra RUNTIME DESTINATION ${bindir})
