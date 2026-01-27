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

    juce::TextButton saveDatasetButton { "Save Dataset" };
    std::unique_ptr<juce::FileChooser> saveChooser;

    juce::TextButton batchButton { "Batch Analyze" };
    std::unique_ptr<juce::FileChooser> batchChooser;

    juce::TextButton loadDatasetButton {"Load Dataset"};
    std::unique_ptr<juce::FileChooser> loadChooser;


    std::array<juce::Slider, 16> pitchSliders;



    juce::Slider toleranceSlider;
    juce::Label toleranceLabel;

    juce::Slider grooveAmountSlider;
    juce::Label grooveLabel;

    juce::ProgressBar trainingProgressBar;
    double progress;

    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> toleranceAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> grooveAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};