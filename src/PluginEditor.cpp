#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p), trainingProgressBar(progress)
{
    // 1. Tools Menu
    addAndMakeVisible(toolsMenu);
    toolsMenu.addItem("Tools", 1);
    toolsMenu.addSeparator();
    toolsMenu.addItem("Batch Analyze", 2);
    toolsMenu.addItem("Train Model", 3);
    toolsMenu.addItem("Save Dataset", 4);
    toolsMenu.addItem("Load Dataset", 5);
    toolsMenu.setSelectedId(1);
    
    toolsMenu.onChange = [this] {
        int id = toolsMenu.getSelectedId();
        if (id == 2) { // Batch Analyze
            batchChooser = std::make_unique<juce::FileChooser> (
                "Select folder of loops...",
                juce::File::getSpecialLocation(juce::File::userHomeDirectory),
                "*.wav");
            batchChooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories,
                [this] (const juce::FileChooser& fc) {
                    auto result = fc.getResult();
                    if (result.isDirectory()) processorRef.triggerBatchAnalysis(result);
                });
        }
        else if (id == 3) { // Train
            processorRef.startTrainingSession(100, 0.001);
        }
        else if (id == 4) { // Save
            saveChooser = std::make_unique<juce::FileChooser> ("Save Dataset...", juce::File(), "*.pt");
            saveChooser->launchAsync (juce::FileBrowserComponent::saveMode, [this] (const juce::FileChooser& fc) {
                auto file = fc.getResult();
                if (file != juce::File()) processorRef.saveDataset(file);
            });
        }
        else if (id == 5) { // Load
            loadChooser = std::make_unique<juce::FileChooser> ("Load Dataset...", juce::File(), "*.pt");
            loadChooser->launchAsync(juce::FileBrowserComponent::openMode, [this] (const juce::FileChooser& fc) {
                auto file = fc.getResult();
                if (file.existsAsFile()) {
                    processorRef.loadDataset(file);
                    bakeHeatmaps(); // Re-bake when new data arrives
                }
            });
        }
        toolsMenu.setSelectedId(1, juce::dontSendNotification);
    };

    // 2. Latent Pads
    padA = std::make_unique<LatentXYPad>(processorRef.apvts, "latent0", "latent1", [this]{
        processorRef.updateModelFromLatent();
    });
    addAndMakeVisible(*padA);

    padB = std::make_unique<LatentXYPad>(processorRef.apvts, "latent2", "latent3", [this]{
        processorRef.updateModelFromLatent();
    });
    addAndMakeVisible(*padB);

    // 3. Generate button
    addAndMakeVisible(generateButton);
    generateButton.onClick = [this] {
        processorRef.generateNewRhythm();
    };

    // Pitch Sliders
    for (int i = 0; i < 16; ++i)
    {
        addAndMakeVisible(pitchSliders[i]);
        pitchSliders[i].setSliderStyle(juce::Slider::LinearVertical);
        pitchSliders[i].setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);

        pitchAttachments[i] = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
            processorRef.apvts, "pitch" + juce::String(i), pitchSliders[i]);
    }

    // Progress Bar
    addAndMakeVisible(trainingProgressBar);

    // Tolerance slider
    addAndMakeVisible(toleranceSlider);
    toleranceSlider.setRange(0.0, 1.0, 0.01);
    toleranceAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processorRef.apvts, "tolerance", toleranceSlider);
    
    addAndMakeVisible(toleranceLabel);
    toleranceLabel.setText("Tolerance", juce::dontSendNotification);
    toleranceLabel.attachToComponent(&toleranceSlider, true);

    // groove slider
    addAndMakeVisible(grooveAmountSlider);
    grooveAmountSlider.setRange(0.0, 1.0, 0.01);
    grooveAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processorRef.apvts, "grooveAmount", grooveAmountSlider);

    addAndMakeVisible(grooveLabel);
    grooveLabel.setText("Groove Amount", juce::dontSendNotification);
    grooveLabel.attachToComponent(&grooveAmountSlider, true);
    
    setSize (1000, 800); 
    startTimerHz(30);

    // Initial bake if data exists
    bakeHeatmaps();
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
    stopTimer();
}

juce::Rectangle<int> AudioPluginAudioProcessorEditor::getStepColumnBounds(int index)
{
    auto area = getLocalBounds().reduced(20);
    area.removeFromTop(600); // Latent Pads + Performance Sliders
    auto sequencerArea = area.removeFromTop(150);
    
    int columnWidth = sequencerArea.getWidth() / 16;
    return sequencerArea.removeFromLeft(columnWidth * (index + 1)).removeFromRight(columnWidth);
}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colours::darkblue);
    
    auto area = getLocalBounds();
    auto stepAreaWidth = area.getWidth() - 80;
    auto stepWidth = stepAreaWidth / 16;
    auto stepHeight = 30;
    auto yPos = 350;
    
    const auto& probs = processorRef.getProbabilities();
    float currentTol = toleranceSlider.getValue();

//    // Get the current rhythm pattern from the processor
//    const auto& rhythm = processorRef.getRhythmArray();

    for (int i = 0; i < 16; ++i)
        {
            auto xPos = 40 + (i * stepWidth);
            bool isCurrentStep = (static_cast<int>(currentStep) % 16 == i);
            
            // REAL-TIME VISUAL THRESHOLDING
            float prob = probs[i].load();
            bool isTriggerActive = (prob > currentTol);

            if (isCurrentStep)
                g.setColour(juce::Colours::red);
            else if (isTriggerActive)
                g.setColour(juce::Colours::white.withAlpha(prob)); // Optional: Dimmer if lower prob
            else
                g.setColour(juce::Colours::darkgrey.darker());

            g.fillRect(xPos, yPos, stepWidth - 5, stepHeight);
        }
}

void AudioPluginAudioProcessorEditor::resized()
{
    auto area = getLocalBounds().reduced(20);
    
    // 1. Top Bar: Tools Menu and Generate Button
    auto topBar = area.removeFromTop(40);
    toolsMenu.setBounds(topBar.removeFromLeft(150));
    generateButton.setBounds(topBar.removeFromRight(150));
    
    // 2. Latent Pads Section
    auto padArea = area.removeFromTop(250);
    auto padWidth = padArea.getWidth() / 2 - 10;
    padA->setBounds(padArea.removeFromLeft(padWidth));
    padArea.removeFromLeft(20); // Gap
    padB->setBounds(padArea.removeFromLeft(padWidth));
    
    // 3. Sliders (Tolerance, Groove)
    area.removeFromTop(10); // Gap
    auto sliderControlArea = area.removeFromTop(40);
    toleranceSlider.setBounds(sliderControlArea.removeFromLeft(300).reduced(40, 0));
    grooveAmountSlider.setBounds(sliderControlArea.removeFromRight(300).reduced(40, 0));

    // 4. Pitch Sliders
    auto pitchArea = area.removeFromTop(100);
    auto stepWidth = pitchArea.getWidth() / 16;
    for (int i = 0; i < 16; ++i)
    {
        pitchSliders[i].setBounds(pitchArea.removeFromLeft(stepWidth).reduced(2, 0));
    }
    
    // 5. Bottom: Progress Bar
    trainingProgressBar.setBounds(area.removeFromBottom(25));
}

void AudioPluginAudioProcessorEditor::bakeHeatmaps()
{
    const int padSize = 250;
    heatmapA = juce::Image(juce::Image::RGB, padSize, padSize, true);
    heatmapB = juce::Image(juce::Image::RGB, padSize, padSize, true);

    float mu[4] = {0,0,0,0};
    float sigma[4] = {1,1,1,1};

    if (processorRef.isDensityEstimated())
    {
        auto means = processorRef.getLatentMeans();
        auto stds = processorRef.getLatentStdDevs();
        
        if (means.numel() == 4 && stds.numel() == 4)
        {
            auto mPtr = means.data_ptr<float>();
            auto sPtr = stds.data_ptr<float>();
            for (int i = 0; i < 4; ++i) {
                mu[i] = mPtr[i];
                sigma[i] = sPtr[i];
            }
        }
    }

    auto renderHeatmap = [&](juce::Image& img, int dimX, int dimY) {
        juce::Image::BitmapData data(img, juce::Image::BitmapData::writeOnly);
        
        for (int y = 0; y < padSize; ++y) {
            for (int x = 0; x < padSize; ++x) {
                float valX = juce::jmap((float)x, 0.0f, (float)padSize, -3.0f, 3.0f);
                float valY = juce::jmap((float)y, 0.0f, (float)padSize, 3.0f, -3.0f); 

                float density = gaussian2D(valX, valY, mu[dimX], mu[dimY], sigma[dimX], sigma[dimY]);
                
                juce::uint8 rVal = (juce::uint8)(juce::jlimit(0.0f, 1.0f, density) * 255);
                juce::uint8 bVal = (juce::uint8)(juce::jlimit(0.0f, 1.0f, 1.0f - density) * 150 + 50);
                juce::uint8 gVal = (juce::uint8)(density * 50);
                
                data.setPixelColour(x, y, juce::Colour(rVal, gVal, bVal));
            }
        }
    };

    renderHeatmap(heatmapA, 0, 1);
    renderHeatmap(heatmapB, 2, 3);

    if (padA) padA->setHeatmap(heatmapA);
    if (padB) padB->setHeatmap(heatmapB);
}

float AudioPluginAudioProcessorEditor::gaussian2D(float x, float y, float muX, float muY, float sigmaX, float sigmaY)
{
    float dx = (x - muX) / sigmaX;
    float dy = (y - muY) / sigmaY;
    return std::exp(-0.5f * (dx * dx + dy * dy));
}

void AudioPluginAudioProcessorEditor::timerCallback()
{
    // Update the local currentStep from the processor
    currentStep = processorRef.getCurrentStep().load();
    
    // Update label text
    stepLabel.setText("Step: " + juce::String((static_cast<int>(currentStep)% 16) + 1), juce::dontSendNotification);

    progress = processorRef.getBackgroundProgress();

    // Trigger heatmap re-bake if training just finished
    if (processorRef.hasFinishedTraining() && processorRef.isDensityEstimated()) {
        bakeHeatmaps();
        // The processor should probably reset finishedTraining or we should track it locally
        // to avoid constant re-baking. For now, bakeHeatmaps is relatively fast.
    }

    repaint();
}
