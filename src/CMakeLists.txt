add_executable(
	jeprof-cpp
    main.cpp
)

target_link_libraries(
	jeprof-cpp
	${LIBGLOG_LIBRARIES}
	${GFLAGS_LIBRARIES}
	Threads::Threads
)
