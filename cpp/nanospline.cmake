if (TARGET nanospline::nanospline)
    return()
endif()

message(STATUS "Third-party (external): creating target 'nanospline::nanospline'")

include(CPM)
CPMAddPackage(
    NAME nanospline
    GITHUB_REPOSITORY qnzhou/nanospline
    GIT_TAG        5657b8d39178e81ebf9ac91b37b47516ec361185
    OPTIONS
    "NANOSPLINE_BUILD_TESTS Off"
)
FetchContent_MakeAvailable(nanospline)

set_target_properties(nanospline PROPERTIES FOLDER third_party)