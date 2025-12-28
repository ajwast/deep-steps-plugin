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
    juce::TextButton trainButton { "Train Model" };
    juce::TextButton importButton { "Import CSV" };
    juce::TextButton analyzeButton { "Analyze Loop" };
    std::unique_ptr<juce::FileChooser> csvChooser;   // Was 'chooser'
    std::unique_ptr<juce::FileChooser> audioChooser;
    
    std::array<juce::Slider, 16> pitchSliders;
    juce::Slider toleranceSlider;
    juce::Label toleranceLabel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};
