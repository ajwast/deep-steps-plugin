#include "PluginProcessor.h"
#include "PluginEditor.h"

static juce::String getMidiNoteName (int noteNumber)
{
    static const char* noteNames[] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
    int octave = (noteNumber / 12) - 1; // standard MIDI mapping (60 is C4)
    return juce::String (noteNames[noteNumber % 12]) + juce::String (octave);
}

//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p), trainingProgressBar(progress)
{
    // Apply look and feel
    setLookAndFeel(&customLookAndFeel);

    // 0. Title Label
    addAndMakeVisible(titleLabel);
    titleLabel.setText("DEEP STEPS", juce::dontSendNotification);
    titleLabel.setFont(customLookAndFeel.getCustomFont(28.0f, true));
    titleLabel.setJustificationType(juce::Justification::centred);
    titleLabel.setColour(juce::Label::textColourId, juce::Colour(0xff00ffcc)); // retro neon cyan

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
            ""); // <-- CHANGED from "*.wav" to "" to fix macOS folder selection

        auto flags = juce::FileBrowserComponent::openMode |
                     juce::FileBrowserComponent::canSelectDirectories;

        batchChooser->launchAsync (flags, [this] (const juce::FileChooser& fc) {
                auto result = fc.getResult();
                if (result.isDirectory()) processorRef.triggerBatchAnalysis(result);
            });
    }
    else if (id == 3) { // Train
        processorRef.startTrainingSession(100, 0.001);
    }
    else if (id == 4) { // Save Dataset
        saveChooser = std::make_unique<juce::FileChooser> ("Save Dataset...", juce::File(), "*.pt");

        // <-- ADDED canSelectFiles flag
        auto flags = juce::FileBrowserComponent::saveMode |
                     juce::FileBrowserComponent::canSelectFiles;

        saveChooser->launchAsync (flags, [this] (const juce::FileChooser& fc) {
            auto file = fc.getResult();
            if (file != juce::File()) processorRef.saveDataset(file);
        });
    }
    else if (id == 5) { // Load Dataset
        loadChooser = std::make_unique<juce::FileChooser> ("Load Dataset...", juce::File(), "*.pt");

        // <-- ADDED canSelectFiles flag
        auto flags = juce::FileBrowserComponent::openMode |
                     juce::FileBrowserComponent::canSelectFiles;

        loadChooser->launchAsync(flags, [this] (const juce::FileChooser& fc) {
            auto file = fc.getResult();
            if (file.existsAsFile()) {
                processorRef.loadDataset(file);
                bakeHeatmaps(); // Re-bake when new data arrives
            }
        });
    }

    // Reset the menu text back to "Tools" so it acts like a dropdown button
    toolsMenu.setSelectedId(1, juce::dontSendNotification);
};

    // 2. Latent Pads with theme coloring
    padA = std::make_unique<LatentXYPad>(processorRef.apvts, "latent0", "latent1", [this]{
        processorRef.updateModelFromLatent();
    }, juce::Colour(0xff00ffcc)); // Retro neon cyan
    addAndMakeVisible(*padA);

    padB = std::make_unique<LatentXYPad>(processorRef.apvts, "latent2", "latent3", [this]{
        processorRef.updateModelFromLatent();
    }, juce::Colour(0xffff007f)); // Retro neon magenta
    addAndMakeVisible(*padB);

    // 3. Generate button
    addAndMakeVisible(generateButton);
    generateButton.onClick = [this] {
        processorRef.generateNewRhythm();
    };

    // Pitch Sliders (Faders)
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

    // Tolerance rotary knob
    addAndMakeVisible(toleranceSlider);
    toleranceSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    toleranceSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    toleranceSlider.setRange(0.0, 1.0, 0.01);
    toleranceAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processorRef.apvts, "tolerance", toleranceSlider);
    
    addAndMakeVisible(toleranceLabel);
    toleranceLabel.setText("Tolerance", juce::dontSendNotification);
    toleranceLabel.setJustificationType(juce::Justification::centred);
    toleranceLabel.setFont(customLookAndFeel.getCustomFont(14.0f, false));

    // Groove Amount rotary knob
    addAndMakeVisible(grooveAmountSlider);
    grooveAmountSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    grooveAmountSlider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    grooveAmountSlider.setRange(0.0, 1.0, 0.01);
    grooveAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        processorRef.apvts, "grooveAmount", grooveAmountSlider);

    addAndMakeVisible(grooveLabel);
    grooveLabel.setText("Groove Amount", juce::dontSendNotification);
    grooveLabel.setJustificationType(juce::Justification::centred);
    grooveLabel.setFont(customLookAndFeel.getCustomFont(14.0f, false));
    
    stepLabel.setFont(customLookAndFeel.getCustomFont(14.0f, false));
    
    setSize (1000, 800); 
    startTimerHz(30);

    // Initial bake if data exists
    bakeHeatmaps();
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
    setLookAndFeel(nullptr);
    stopTimer();
}

juce::Rectangle<int> AudioPluginAudioProcessorEditor::getStepColumnBounds(int index)
{
    auto area = getLocalBounds().reduced(20, 0); // 960 width
    auto sequencerArea = juce::Rectangle<int>(area.getX(), 530, area.getWidth(), 230);
    
    int columnWidth = sequencerArea.getWidth() / 16;
    return sequencerArea.removeFromLeft(columnWidth * (index + 1)).removeFromRight(columnWidth);
}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g)
{
    // 1. Background (Dark retro navy/slate)
    g.fillAll (juce::Colour(0xff1e2230));
    
    // 2. Header Background (with bottom neon-cyan separator)
    g.setColour(juce::Colour(0xff161822));
    g.fillRect(0, 0, getWidth(), 80);
    
    g.setColour(juce::Colour(0xff00ffcc).withAlpha(0.35f));
    g.drawHorizontalLine(79, 0.0f, (float)getWidth());

    // 3. Sequencer Card/Section Panel
    auto sequencerBounds = juce::Rectangle<int>(20, 530, getWidth() - 40, 230);
    g.setColour(juce::Colour(0xff161822));
    g.fillRoundedRectangle(sequencerBounds.toFloat(), 4.0f);
    g.setColour(juce::Colour(0xff2a2d3d));
    g.drawRoundedRectangle(sequencerBounds.toFloat(), 4.0f, 1.2f);

    // 4. Sequencer Step Indicators
    const auto& probs = processorRef.getProbabilities();
    float currentTol = toleranceSlider.getValue();
    int activeStep = static_cast<int>(currentStep) % 16;

    for (int i = 0; i < 16; ++i)
    {
        auto colBounds = getStepColumnBounds(i);
        
        // Draw column separator
        g.setColour(juce::Colours::white.withAlpha(0.04f));
        g.drawVerticalLine(colBounds.getX(), (float)colBounds.getY(), (float)colBounds.getBottom());

        // Step Indicator (Square light)
        auto lightBounds = colBounds.removeFromTop(30).reduced(5, 5);
        float prob = probs[i].load();
        bool isTriggerActive = (prob > currentTol);
        auto cornerSize = 2.0f;

        if (i == activeStep)
        {
            // Active step indicator: Glowing orange square
            g.setColour(juce::Colour(0xffff9f0a));
            g.fillRoundedRectangle(lightBounds.toFloat(), cornerSize);
            
            g.setColour(juce::Colour(0xffff9f0a).withAlpha(0.4f));
            g.drawRoundedRectangle(lightBounds.toFloat().expanded(1.5f), cornerSize, 1.5f);
        }
        else if (isTriggerActive)
        {
            // Predicted trigger step: Cyan square scaled by probability
            g.setColour(juce::Colour(0xff00ffcc).withAlpha(prob));
            g.fillRoundedRectangle(lightBounds.toFloat(), cornerSize);
        }
        else
        {
            // Dark inactive step square
            g.setColour(juce::Colour(0xff12131a));
            g.fillRoundedRectangle(lightBounds.toFloat(), cornerSize);
            g.setColour(juce::Colour(0xff2d3142));
            g.drawRoundedRectangle(lightBounds.toFloat(), cornerSize, 1.0f);
        }

        // Draw note readout text at the bottom of the column strip
        auto textBounds = colBounds.removeFromBottom(20);
        int noteNumber = static_cast<int>(pitchSliders[i].getValue());
        g.setColour(juce::Colour(0xffe2e8f0));
        g.setFont(customLookAndFeel.getCustomFont(11.0f, false));
        g.drawFittedText(getMidiNoteName(noteNumber), textBounds, juce::Justification::centred, 1);

        // Highlight column under current playback step
        if (i == activeStep)
        {
            g.setColour(juce::Colour(0xffff9f0a).withAlpha(0.03f));
            g.fillRect(colBounds);
        }
    }
}

void AudioPluginAudioProcessorEditor::resized()
{
    auto area = getLocalBounds();
    
    // 1. Header (80px)
    auto headerArea = area.removeFromTop(80).reduced(20, 0);
    titleLabel.setBounds(headerArea.getX(), headerArea.getY() + 10, headerArea.getWidth(), 30);
    toolsMenu.setBounds(headerArea.getX(), headerArea.getY() + 45, 200, 30);
    generateButton.setBounds(headerArea.getRight() - 200, headerArea.getY() + 45, 200, 30);
    
    // 2. Latent Performance (360px)
    auto padArea = area.removeFromTop(360).reduced(20, 0);
    int padSize = 340;
    padA->setBounds(padArea.removeFromLeft(padSize).reduced(0, 10));
    padArea.removeFromLeft(padArea.getWidth() - padSize); // Skip center gap
    padB->setBounds(padArea.removeFromLeft(padSize).reduced(0, 10));
    
    // 3. Performance Knobs (90px)
    auto sliderControlArea = area.removeFromTop(90).reduced(20, 0);
    int centerX = sliderControlArea.getCentreX();
    int knobSize = 60;
    int labelHeight = 20;
    int centerGap = 160;
    
    int toleranceX = centerX - knobSize - centerGap / 2;
    int toleranceY = sliderControlArea.getY();
    toleranceLabel.setBounds(toleranceX - 25, toleranceY, knobSize + 50, labelHeight);
    toleranceSlider.setBounds(toleranceX, toleranceY + labelHeight, knobSize, knobSize);
    
    int grooveX = centerX + centerGap / 2;
    int grooveY = sliderControlArea.getY();
    grooveLabel.setBounds(grooveX - 25, grooveY, knobSize + 50, labelHeight);
    grooveAmountSlider.setBounds(grooveX, grooveY + labelHeight, knobSize, knobSize);

    // 4. Sequencer Area (230px)
    auto sequencerArea = area.removeFromTop(230);
    for (int i = 0; i < 16; ++i)
    {
        auto colBounds = getStepColumnBounds(i);
        colBounds.removeFromTop(35); // Space for Step Light (drawn in paint)
        colBounds.removeFromBottom(20); // Space for Note Readout (drawn in paint)
        pitchSliders[i].setBounds(colBounds.reduced(2, 5));
    }
    
    // 5. Bottom: Progress Bar & Footer Info
    auto footerArea = area.reduced(20, 0);
    trainingProgressBar.setBounds(footerArea.removeFromLeft(800).reduced(0, 10));
    stepLabel.setBounds(footerArea.reduced(0, 10));
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

    auto renderHeatmap = [&](juce::Image& img, int dimX, int dimY, juce::Colour hotColor) {
        juce::Image::BitmapData data(img, juce::Image::BitmapData::writeOnly);

        // Define the chassis background color (Cold)
        juce::Colour coldColor = juce::Colour(0xff12131a);

        for (int y = 0; y < padSize; ++y) {
            for (int x = 0; x < padSize; ++x) {
                float valX = juce::jmap((float)x, 0.0f, (float)padSize, -3.0f, 3.0f);
                float valY = juce::jmap((float)y, 0.0f, (float)padSize, 3.0f, -3.0f);

                float density = gaussian2D(valX, valY, mu[dimX], mu[dimY], sigma[dimX], sigma[dimY]);
                density = juce::jlimit(0.0f, 1.0f, density);

                // Interpolate smoothly between chassis color and neon accent
                juce::Colour pixelColor = coldColor.interpolatedWith(hotColor, density);
                data.setPixelColour(x, y, pixelColor);
            }
        }
    };

    // Now pass the specific neon colors for Pad A and Pad B when you call it
    renderHeatmap(heatmapA, 0, 1, juce::Colour(0xff00ffcc)); // Neon Cyan
    renderHeatmap(heatmapB, 2, 3, juce::Colour(0xffff007f)); // Neon Magenta

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
