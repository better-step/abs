if (TARGET nanospline::nanospline)
    return()
endif()

message(STATUS "Third-party (external): creating target 'nanospline::nanospline'")

include(CPM)
CPMAddPackage(
    NAME nanospline
    GITHUB_REPOSITORY qnzhou/nanospline
    GIT_TAG        660e3db75d8faa43f201f7638e9bd198bd5237d5
    OPTIONS
    "NANOSPLINE_BUILD_TESTS Off"
)
FetchContent_MakeAvailable(nanospline)

set_target_properties(nanospline PROPERTIES FOLDER third_party)