#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p), trainingProgressBar(progress)
{

    // Generate button
    addAndMakeVisible(generateButton);
    generateButton.onClick = [this] {
        processorRef.generateNewRhythm();
        repaint(); // Force a redraw so we see the new pattern immediately
    };
    


    // Pitch Sliders
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

    // Batch button
    addAndMakeVisible(batchButton);
    batchButton.onClick = [this] {
        batchChooser = std::make_unique<juce::FileChooser> (
            "Select folder of loops...",
            juce::File::getSpecialLocation(juce::File::userHomeDirectory),
            "*.wav");

        batchChooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories,
            [this] (const juce::FileChooser& fc)
            {
                auto result = fc.getResult();
                if (result.isDirectory())
                {
                    // This call returns instantly, keeping the UI snappy
                    processorRef.triggerBatchAnalysis(result);
                }
            });
    };


    addAndMakeVisible(trainingProgressBar);

    // Train button
    addAndMakeVisible(trainButton);
    trainButton.onClick = [this] {
        processorRef.startTrainingSession(200, 0.001);
    };

    // Tolerance slider
    addAndMakeVisible(toleranceSlider);
    toleranceSlider.setRange(0.0, 1.0, 0.01);
    toleranceSlider.setValue(processorRef.tolerance.load());
    toleranceSlider.onValueChange = [this] {
        processorRef.tolerance.store((float)toleranceSlider.getValue());
        repaint(); // Update UI immediately when sliding
    };
    // Tolerance label
    addAndMakeVisible(toleranceLabel);
    toleranceLabel.setText("Tolerance", juce::dontSendNotification);
    toleranceLabel.attachToComponent(&toleranceSlider, true);

    // groove slider
    addAndMakeVisible(grooveAmountSlider);
    grooveAmountSlider.setRange(0.0, 1.0, 0.01);
    grooveAmountSlider.setValue(processorRef.grooveAmount.load());
    grooveAmountSlider.onValueChange = [this] {
        processorRef.grooveAmount.store((float)grooveAmountSlider.getValue());
    };
    // Groove label
    addAndMakeVisible(grooveLabel);
    grooveLabel.setText("Groove Amount", juce::dontSendNotification);
    grooveLabel.attachToComponent(&grooveAmountSlider, true);
    
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

    // 1. Arrange buttons in a column on the left
    generateButton.setBounds(topArea.removeFromLeft(200).withHeight(buttonHeight).withY(20));
    trainButton.setBounds(generateButton.getBounds().translated(0, 40));
    saveDatasetButton.setBounds(trainButton.getBounds().translated(0, 40));

    // Bottom button
    batchButton.setBounds(area.removeFromBottom(40).withSize(150, 30));

    trainingProgressBar.setBounds(area.removeFromBottom(25));

    // 2. Control Sliders on the right side
    // Position Tolerance Slider
    toleranceSlider.setBounds(topArea.withSize(250, buttonHeight).withPosition(450, 20));

    // Position Groove Slider (shifted 40 pixels down from the Tolerance slider)
    grooveAmountSlider.setBounds(toleranceSlider.getBounds().translated(0, 40));

    // 3. Pitch Sliders in the middle
    auto sliderArea = area.removeFromTop(150);
    auto stepWidth = sliderArea.getWidth() / 16;
    for (int i = 0; i < 16; ++i)
    {
        pitchSliders[i].setBounds(sliderArea.removeFromLeft(stepWidth).reduced(2, 0));
    }
}

void AudioPluginAudioProcessorEditor::timerCallback()
{
    // Update the local currentStep from the processor
    currentStep = processorRef.getCurrentStep().load();
    
    // Update label text
    stepLabel.setText("Step: " + juce::String((static_cast<int>(currentStep)% 16) + 1), juce::dontSendNotification);

    progress = processorRef.getBackgroundProgress();

    // TRIGGER THE UI REFRESH
    repaint();
}
