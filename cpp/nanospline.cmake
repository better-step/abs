if (TARGET nanospline::nanospline)
    return()
endif()

message(STATUS "Third-party (external): creating target 'nanospline::nanospline'")

include(CPM)
CPMAddPackage(
    NAME nanospline
    GITHUB_REPOSITORY teseoch/nanospline
    GIT_TAG        a4bba801c09177f2beaf0522554e4fdf7a49ce1a
    OPTIONS
    "NANOSPLINE_BUILD_TESTS Off"
    "NANOSPLINE_MSHIO Off"
)
FetchContent_MakeAvailable(nanospline)

set_target_properties(nanospline PROPERTIES FOLDER third_party)