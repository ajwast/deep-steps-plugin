#include "PluginProcessor.h"
#include "LatentXYPad.h"
#include "CustomLookAndFeel.h"

//==============================================================================
class AudioPluginAudioProcessorEditor final : public juce::AudioProcessorEditor, 
                                              private juce::Timer
{
public:
    explicit AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor&);
    ~AudioPluginAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

    void bakeHeatmaps();

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    AudioPluginAudioProcessor& processorRef;

    CustomLookAndFeel customLookAndFeel;
    juce::Label titleLabel;
    juce::Label stepLabel;
    double currentStep;

    // --- NEW UI ELEMENTS ---
    juce::ComboBox toolsMenu;
    std::unique_ptr<LatentXYPad> padA, padB;
    juce::Image heatmapA, heatmapB;

    juce::TextButton generateButton { "Random Latent" };

    // File choosers (keep these)
    std::unique_ptr<juce::FileChooser> saveChooser;
    std::unique_ptr<juce::FileChooser> batchChooser;
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
    std::array<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>, 16> pitchAttachments;

    float gaussian2D(float x, float y, float muX, float muY, float sigmaX, float sigmaY);

    juce::Rectangle<int> getStepColumnBounds(int index);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};