cmake_minimum_required(VERSION 3.22)
project(PICOPerformanceTest C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- PICO SDK パス ----
# PICO%20OpenXR%20SDK の %20 はリテラル文字のため、CMake の EXISTS では正常に解決される
# ビルド前に subst P: ... を実行してある場合は P: が優先される
set(_PICO_OPENXR_DIRECT
    "C:/Users/halby/AppData/Roaming/PICO_Developer_Center_Downloads/global/sdk_id_11/PICO%20OpenXR%20SDK%203.0.0-20250422/OpenXR_Native_SDK")

if(EXISTS "P:/OpenXR/openxr/openxr.h")
    set(PICO_OPENXR_ROOT "P:")
    message(STATUS "Using P: drive for OpenXR SDK")
elseif(EXISTS "${_PICO_OPENXR_DIRECT}/OpenXR/openxr/openxr.h")
    set(PICO_OPENXR_ROOT "${_PICO_OPENXR_DIRECT}")
    message(STATUS "Using direct path for OpenXR SDK")
else()
    message(FATAL_ERROR
        "OpenXR SDK not found. Please run:\n"
        "  subst P: \"C:\\Users\\halby\\AppData\\Roaming\\PICO_Developer_Center_Downloads\\global\\sdk_id_11\\PICO%%20OpenXR%%20SDK%%203.0.0-20250422\\OpenXR_Native_SDK\"\n"
        "or verify the SDK installation path.")
endif()

set(OPENXR_INCLUDE_DIR "${PICO_OPENXR_ROOT}/OpenXR")
set(OPENXR_LOADER "${PICO_OPENXR_ROOT}/OpenXR/pico/loader/lib/${ANDROID_ABI}/libopenxr_loader.so")
set(GLM_INCLUDE_DIR "C:/PICO_SDK/3rdParty/glm-0.9.9.8")

# ---- glslc（ホスト PC 上で動作するシェーダーコンパイラ）----
find_program(GLSLC glslc
    HINTS
        "C:/Users/halby/AppData/Local/Android/Sdk/ndk/27.2.12479018/shader-tools/windows-x86_64"
        "C:/Users/halby/AppData/Local/Android/Sdk/ndk/27.1.12297006/shader-tools/windows-x86_64"
        "C:/Users/halby/AppData/Local/Android/Sdk/ndk/27.0.12077973/shader-tools/windows-x86_64"
    REQUIRED)
message(STATUS "glslc: ${GLSLC}")

# ---- SPIR-V → C++ ヘッダーへ変換するカスタムコマンド ----
set(SHADER_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders_gen")
file(MAKE_DIRECTORY "${SHADER_GEN_DIR}")

function(embed_shader SOURCE_GLSL VAR_NAME)
    set(SPV_FILE  "${SHADER_GEN_DIR}/${VAR_NAME}.spv")
    set(HDR_FILE  "${SHADER_GEN_DIR}/${VAR_NAME}.h")
    add_custom_command(
        OUTPUT "${HDR_FILE}"
        COMMAND "${GLSLC}" --target-env=vulkan1.0 -o "${SPV_FILE}" "${SOURCE_GLSL}"
        COMMAND "${CMAKE_COMMAND}"
            "-DINPUT=${SPV_FILE}"
            "-DOUTPUT=${HDR_FILE}"
            "-DVAR_NAME=${VAR_NAME}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
        DEPENDS "${SOURCE_GLSL}"
        COMMENT "Compiling shader ${SOURCE_GLSL} -> ${HDR_FILE}"
        VERBATIM
    )
endfunction()

embed_shader("${CMAKE_CURRENT_SOURCE_DIR}/shaders/vertex.vert" vertex_spv)
embed_shader("${CMAKE_CURRENT_SOURCE_DIR}/shaders/fragment.frag" fragment_spv)
embed_shader("${CMAKE_CURRENT_SOURCE_DIR}/shaders/culling.comp" culling_spv)

# ---- android_native_app_glue（静的ライブラリ）----
add_library(app_glue STATIC
    ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)
target_include_directories(app_glue PUBLIC
    ${ANDROID_NDK}/sources/android/native_app_glue)
target_link_libraries(app_glue PUBLIC android log)

# ---- メインターゲット ----
add_library(PICOPerformanceTest MODULE
    cpp/main.cpp
    # シェーダーヘッダーを依存として追加（カスタムコマンドを確実にトリガーする）
    "${SHADER_GEN_DIR}/vertex_spv.h"
    "${SHADER_GEN_DIR}/fragment_spv.h"
    "${SHADER_GEN_DIR}/culling_spv.h"
)

target_include_directories(PICOPerformanceTest PRIVATE
    "${OPENXR_INCLUDE_DIR}"
    "${GLM_INCLUDE_DIR}"
    "${SHADER_GEN_DIR}"
    ${ANDROID_NDK}/sources/android/native_app_glue
)

target_compile_definitions(PICOPerformanceTest PRIVATE
    XR_USE_PLATFORM_ANDROID
    XR_USE_GRAPHICS_API_VULKAN
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_FORCE_CTOR_INIT
)

target_compile_options(PICOPerformanceTest PRIVATE
    -Wall -Wextra
    -Wno-unused-parameter
    -Wno-missing-field-initializers
    -fno-stack-protector
)

target_link_libraries(PICOPerformanceTest PRIVATE
    android
    app_glue
    log
    vulkan
    "${OPENXR_LOADER}"
)

set_target_properties(PICOPerformanceTest PROPERTIES
    LINK_FLAGS "-u ANativeActivity_onCreate")
