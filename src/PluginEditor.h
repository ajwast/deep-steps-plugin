#pragma once

#include "PluginProcessor.h"

//==============================================================================
class AudioPluginAudioProcessorEditor final : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor&);
    ~AudioPluginAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;
private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    AudioPluginAudioProcessor& processorRef;
    
    juce::Label stepLabel;
    double currentStep;

    juce::TextButton generateButton { "Generate Rhythm" };
    std::array<juce::Slider, 16> pitchSliders;
    juce::TextButton trainButton { "Train Model" };
    juce::TextButton importButton { "Import Data" };
    std::unique_ptr<juce::FileChooser> chooser;
    
    juce::Slider toleranceSlider; // Added
    juce::Label toleranceLabel;   // Added
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};
