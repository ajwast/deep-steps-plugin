cmake_minimum_required(VERSION 3.22)

# 0. System Configuration
# This MUST be set before project() to ensure all targets use it
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "Minimum macOS version")
endif()

# 1. Project Setup
project(DeepSteps VERSION 0.0.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. JUCE Setup 
set(JUCE_DIR "/Users/alexwastnidge/JUCE")
add_subdirectory(${JUCE_DIR} "${CMAKE_BINARY_DIR}/JUCE")

# 3. Create the Plugin Target
juce_add_plugin(DeepSteps
    COMPANY_NAME "AW"
    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE       # CHANGED TO TRUE: Required for your sequencer to work!
    IS_MIDI_EFFECT FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
    PLUGIN_MANUFACTURER_CODE AWDS
    PLUGIN_CODE Ds01
    FORMATS AU VST3 Standalone
    PRODUCT_NAME "Deep Steps")

# 4. Organise Files
set(SourceFiles
    src/PluginProcessor.cpp
    src/PluginProcessor.h
    src/PluginEditor.cpp
    src/PluginEditor.h
    src/Autoencoder.h
)

source_group("Source" FILES ${SourceFiles})

# 5. Add Sources to Target
target_sources(DeepSteps
    PRIVATE
        ${SourceFiles})

# 6. LibTorch Setup
set(CMAKE_PREFIX_PATH "../../../libtorch/libtorch")
find_package(Torch REQUIRED)

# 7. Compile Definitions
target_compile_definitions(DeepSteps
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0)

# 8. Linking
target_link_libraries(DeepSteps
    PRIVATE
        juce::juce_audio_utils
        ${TORCH_LIBRARIES}
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)

juce_generate_juce_header(DeepSteps)