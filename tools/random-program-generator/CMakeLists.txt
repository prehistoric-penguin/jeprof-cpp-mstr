add_executable(
	random-program-generator
    main.cpp
)

target_link_libraries(
	random-program-generator
	${FBTHRIFT_LIBRARIES}
	${GFLAGS_LIBRARIES}
	${LIBGLOG_LIBRARIES}
	Threads::Threads
)
