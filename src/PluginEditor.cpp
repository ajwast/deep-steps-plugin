#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p)
{
    // Set up the step label
    addAndMakeVisible(stepLabel);
    stepLabel.setFont(juce::FontOptions(24.0f));
    stepLabel.setJustificationType(juce::Justification::centred);
    stepLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    
    addAndMakeVisible(generateButton);
        
    // Use a lambda to tell the processor to generate a new rhythm
    generateButton.onClick = [this] {
        processorRef.generateNewRhythm();
        repaint(); // Force a redraw so we see the new pattern immediately
    };
    
//    auto& pitchRef = processorRef.getPitchArray();

    for (int i = 0; i < 16; ++i)
        {
            addAndMakeVisible(pitchSliders[i]);
            pitchSliders[i].setSliderStyle(juce::Slider::LinearVertical);
            pitchSliders[i].setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
            pitchSliders[i].setRange(36, 96, 1);
            pitchSliders[i].setValue(processorRef.getPitchArray()[i]);

            pitchSliders[i].onValueChange = [this, i] {
                processorRef.getPitchArray()[i] = static_cast<int>(pitchSliders[i].getValue());
            };
        }

        addAndMakeVisible(importButton); // Ensure this is visible!
        importButton.onClick = [this] {
            csvChooser = std::make_unique<juce::FileChooser> ("Select CSV", juce::File(), "*.csv");
            auto flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;

            csvChooser->launchAsync(flags, [this](const juce::FileChooser& fc) {
                auto file = fc.getResult();
                if (file.existsAsFile())
                    processorRef.loadDataFromFile(file);
            });
        };

        addAndMakeVisible(trainButton);
        trainButton.onClick = [this] {
            processorRef.startTrainingSession(100, 0.001);
        };
    
    addAndMakeVisible(toleranceSlider);
    toleranceSlider.setRange(0.0, 1.0, 0.01);
    toleranceSlider.setValue(processorRef.tolerance.load());
    toleranceSlider.onValueChange = [this] {
        processorRef.tolerance.store((float)toleranceSlider.getValue());
        repaint(); // Update UI immediately when sliding
    };

    addAndMakeVisible(toleranceLabel);
    toleranceLabel.setText("Tolerance", juce::dontSendNotification);
    toleranceLabel.attachToComponent(&toleranceSlider, true);
    
    addAndMakeVisible(analyzeButton);
    analyzeButton.onClick = [this] {
    audioChooser = std::make_unique<juce::FileChooser> (
        "Select an audio file for analysis...",
        juce::File::getSpecialLocation(juce::File::userHomeDirectory),
        "*.wav;*.aif;*.mp3"
    );

    auto flags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;

    audioChooser->launchAsync (flags, [this] (const juce::FileChooser& fc) {
        auto file = fc.getResult();
        if (file.existsAsFile())
            processorRef.loadAudioFile(file);
    });
    };
    
    setSize (800, 500);
    startTimerHz(30); // 30 FPS is usually plenty for a sequencer UI
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
    stopTimer();
}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colours::darkblue);
    
    auto area = getLocalBounds();
    auto stepAreaWidth = area.getWidth() - 80;
    auto stepWidth = stepAreaWidth / 16;
    auto stepHeight = 30;
    auto yPos = 400;
    
    const auto& probs = processorRef.getProbabilities();
    float currentTol = processorRef.tolerance.load();

//    // Get the current rhythm pattern from the processor
//    const auto& rhythm = processorRef.getRhythmArray();

    for (int i = 0; i < 16; ++i)
        {
            auto xPos = 40 + (i * stepWidth);
            bool isCurrentStep = (static_cast<int>(currentStep) % 16 == i);
            
            // REAL-TIME VISUAL THRESHOLDING
            bool isTriggerActive = (probs[i] > currentTol);

            if (isCurrentStep)
                g.setColour(juce::Colours::red);
            else if (isTriggerActive)
                g.setColour(juce::Colours::white.withAlpha(probs[i])); // Optional: Dimmer if lower prob
            else
                g.setColour(juce::Colours::darkgrey.darker());

            g.fillRect(xPos, yPos, stepWidth - 5, stepHeight);
        }
}

void AudioPluginAudioProcessorEditor::resized()
{
    auto area = getLocalBounds().reduced(20);
    auto topArea = area.removeFromTop(150);
    
    auto buttonWidth = 150;
    auto buttonHeight = 30;

    // Arrange buttons in a column on the left
    generateButton.setBounds(topArea.removeFromLeft(200).withHeight(buttonHeight).withY(20));
    trainButton.setBounds(generateButton.getBounds().translated(0, 40));
    importButton.setBounds(trainButton.getBounds().translated(0, 40));

    // Put Tolerance Slider on the right side of the top area
    toleranceSlider.setBounds(topArea.withSize(250, buttonHeight).withPosition(450, 20));

    // Pitch Sliders in the middle
    auto sliderArea = area.removeFromTop(150);
    auto stepWidth = sliderArea.getWidth() / 16;
    for (int i = 0; i < 16; ++i)
    {
        pitchSliders[i].setBounds(sliderArea.removeFromLeft(stepWidth).reduced(2, 0));
    }
    
    analyzeButton.setBounds(10, 10, 150, 30);
}

void AudioPluginAudioProcessorEditor::timerCallback()
{
    // Update the local currentStep from the processor
    currentStep = processorRef.getCurrentStep().load();
    
    // Update label text
    stepLabel.setText("Step: " + juce::String((static_cast<int>(currentStep)% 16) + 1), juce::dontSendNotification);
    
    // TRIGGER THE UI REFRESH
    repaint();
}
