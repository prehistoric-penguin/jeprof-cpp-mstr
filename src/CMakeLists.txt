add_executable(
	jeprof-cpp
    main.cpp
)

target_link_libraries(
	jeprof-cpp
	libprofiler.a
	unwind
	${GFLAGS_LIBRARIES}
  ${LIBGLOG_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_REGEX_LIBRARY}
  ${TCMALLOC_LIBRARIES}
	Threads::Threads
)
