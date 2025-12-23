#include "PluginProcessor.h"
#include "PluginEditor.h"

#include <fstream>

//==============================================================================
AudioPluginAudioProcessor::AudioPluginAudioProcessor()
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ), juce::Thread("TrainingThread")
{
}

AudioPluginAudioProcessor::~AudioPluginAudioProcessor() {
    stopThread(2000);
}

//==============================================================================
// MIDI HELPER FUNCTIONS
//==============================================================================

//void AudioPluginAudioProcessor::makeMIDINote(int noteNumber, int sampleOffset, juce::MidiBuffer& targetBuffer)
//{
//    auto noteOn = juce::MidiMessage::noteOn(1, noteNumber, (juce::uint8)100);
//
//    // Add to the buffer passed from processBlock, NOT the internal member
//    targetBuffer.addEvent(noteOn, sampleOffset);
//
//    // Use the stored sample rate
//    int64_t noteOffSample = blockStartSample + sampleOffset + static_cast<int64_t>(currentSampleRate * noteLengthSeconds);
//    pendingNotes.push_back({noteOn.getChannel(), noteOn.getNoteNumber(), noteOffSample});
//}

void AudioPluginAudioProcessor::makeMIDINote(int noteNumber, int sampleOffset)
{
    auto noteOn = juce::MidiMessage::noteOn(1, noteNumber, (juce::uint8)100);
    midiBuffer.addEvent(noteOn, sampleOffset);

    // Calculate absolute Note Off sample position
    int64_t noteOffSample = blockStartSample + sampleOffset + static_cast<int64_t>(sampleRate * noteLengthSeconds);
    pendingNotes.push_back({noteOn.getChannel(), noteOn.getNoteNumber(), noteOffSample});
}

void AudioPluginAudioProcessor::processPendingNotes(juce::MidiBuffer& targetMidiBuffer, int64_t currentBlockStart, int numSamplesInBlock)
{
    for (auto it = pendingNotes.begin(); it != pendingNotes.end();)
    {
        const auto& [channel, noteNumber, noteOffSample] = *it;

        if (noteOffSample >= currentBlockStart && noteOffSample < currentBlockStart + numSamplesInBlock)
        {
            int sampleOffset = static_cast<int>(noteOffSample - currentBlockStart);
            auto noteOff = juce::MidiMessage::noteOff(channel, noteNumber);
            targetMidiBuffer.addEvent(noteOff, sampleOffset);
            it = pendingNotes.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

//==============================================================================
void AudioPluginAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    this->sampleRate = sampleRate;
    midiBuffer.clear();
    pendingNotes.clear();
    
    currentSampleRate = sampleRate; // Store this for the MIDI math
    blockStartSample = 0;           // Reset timeline
    pendingNotes.clear();
    
    // Default patterns
    rhythmArray.fill(0);
    pitchArray.fill(60);

    // --- Neural Network Inference Test ---
    torch::NoGradGuard no_grad; // Disable gradient calculation for inference
    model.eval();               // Set BatchNorm to evaluation mode

    try {
        torch::Tensor latent = torch::rand({1, 4});
        torch::Tensor generated = model.decode(latent);
        
        auto generatedAccessor = generated.accessor<float, 2>();
        for (int i = 0; i < 16; ++i) {
            rhythmArray[i] = (generatedAccessor[0][i] > tolerance) ? 1 : 0;
        }
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("Torch Error: " + juce::String(e.what()));
    }

    intStep = 0;
    lastStep.store(-1);
}

void AudioPluginAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    auto* playHead = getPlayHead();
    if (playHead == nullptr) return;

    auto optPos = playHead->getPosition();
    if (!optPos.hasValue()) return;

    bpm = optPos->getBpm().orFallback(120.0);
    blockStartSample = optPos->getTimeInSamples().orFallback(0);
    
    numSamples = buffer.getNumSamples();

    // 1. Calculate timing once per block instead of inside the loop if possible
    // But for sample-accurate MIDI triggering, we check samples:
    double samplesPer16th = (sampleRate * 60.0) / (bpm * 4.0);

    for (int sample = 0; sample < numSamples; ++sample)
    {
        double currentSampleAbs = static_cast<double>(blockStartSample + sample);
        double ppqnPosition = currentSampleAbs / samplesPer16th;
        int currentIntStep = static_cast<int>(std::fmod(ppqnPosition, 16.0));

        if (currentIntStep != lastStep.load())
        {
            lastStep.store(currentIntStep);
            currentStep.store(ppqnPosition); // For UI Sync

            if (probabilitiesArray[currentIntStep] > tolerance.load())
            {
                int pitch = pitchArray[currentIntStep];
                makeMIDINote(pitch, sample);
            }
        }
    }

    // 2. Add our generated MIDI to the host's buffer
    midiMessages.addEvents(midiBuffer, 0, numSamples, 0);
    midiBuffer.clear(); // Clear local buffer so we don't double-trigger next block

    // 3. Handle Note Offs
    processPendingNotes(midiMessages, blockStartSample, numSamples);
}


void AudioPluginAudioProcessor::generateNewRhythm()
{
    torch::NoGradGuard no_grad;
    model.eval();

    try {
        // Generate from a random latent vector
        torch::Tensor latent = torch::rand({1, 4});
        torch::Tensor generated = model.decode(latent);
        
        auto generatedAccessor = generated.accessor<float, 2>();

        for (int i = 0; i < 16; ++i) {
            probabilitiesArray[i] = generatedAccessor[0][i];
        }
    } catch (const std::exception& e) {
        juce::Logger::writeToLog("Torch Error during generation: " + juce::String(e.what()));
    }
}

torch::Tensor AudioPluginAudioProcessor::importCsvToTensor(const juce::File& file)
{
    std::vector<float> data;
    std::ifstream stream(file.getFullPathName().toStdString());
    
    if (!stream.is_open()) return torch::Tensor();

    std::string line;
    while (std::getline(stream, line)) {
        std::stringstream ss(line);
        std::string val;
        while (std::getline(ss, val, ',')) {
            try {
                // Basic cleanup of whitespace
                val.erase(std::remove_if(val.begin(), val.end(), ::isspace), val.end());
                if (!val.empty()) data.push_back(std::stof(val));
            } catch (...) {}
        }
    }

    if (data.size() < 16 || data.size() % 16 != 0) {
        juce::Logger::writeToLog("CSV Error: Invalid data size (" + juce::String((int)data.size()) + ")");
        return torch::Tensor();
    }

    int rows = static_cast<int>(data.size() / 16);
    // Clone is essential because the vector 'data' will be destroyed
    return torch::from_blob(data.data(), {rows, 16}, torch::kFloat).clone();
}

void AudioPluginAudioProcessor::loadDataFromFile(const juce::File& file)
{
    // SAFETY: Prevent loading new data while the training thread is active
    if (isThreadRunning()) {
        juce::Logger::writeToLog("Cannot load data while training is in progress.");
        return;
    }

    trainingTensor = importCsvToTensor(file);
    if (trainingTensor.numel() > 0)
        juce::Logger::writeToLog("Data loaded: " + juce::String(trainingTensor.size(0)) + " rows.");
}

void AudioPluginAudioProcessor::startTrainingSession(int epochs, double lr)
{
    if (trainingTensor.numel() > 0 && !isThreadRunning())
    {
        trainingEpochs = epochs;
        trainingLR = lr;
        startThread();
    }
}

void AudioPluginAudioProcessor::run()
{
    if (trainingTensor.numel() == 0) return;

    // FIX: Use the single-tensor constructor.
    // We explicitly tell Stack to expect 'void' as the target type to match the dataset.
    auto dataset = torch::data::datasets::TensorDataset(trainingTensor)
                   .map(torch::data::transforms::Stack<torch::data::Example<torch::Tensor, void>>());
    
    auto dataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset),
        torch::data::DataLoaderOptions(32)); // Batch size 32

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(trainingLR));

    for (int epoch = 0; epoch < trainingEpochs; ++epoch)
    {
        if (threadShouldExit()) return;

        for (auto& batch : *dataLoader)
        {
            // batch.data is now a stacked Tensor of shape [32, 16]
            model.trainBatch(batch.data, optimizer);
        }
    }
    
    model.eval();
    juce::Logger::writeToLog("Background Training Finished!");
}

//==============================================================================
// Standard JUCE boilerplate (Condensed for brevity)
//==============================================================================

const juce::String AudioPluginAudioProcessor::getName() const { return JucePlugin_Name; }
bool AudioPluginAudioProcessor::acceptsMidi() const { return true; }
bool AudioPluginAudioProcessor::producesMidi() const { return true; }
bool AudioPluginAudioProcessor::isMidiEffect() const { return false; }
double AudioPluginAudioProcessor::getTailLengthSeconds() const { return 0.0; }
int AudioPluginAudioProcessor::getNumPrograms() { return 1; }
int AudioPluginAudioProcessor::getCurrentProgram() { return 0; }
void AudioPluginAudioProcessor::setCurrentProgram (int index) {}
const juce::String AudioPluginAudioProcessor::getProgramName (int index) { return {}; }
void AudioPluginAudioProcessor::changeProgramName (int index, const juce::String& newName) {}

void AudioPluginAudioProcessor::releaseResources() { pendingNotes.clear(); }

bool AudioPluginAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;
    return true;
}

bool AudioPluginAudioProcessor::hasEditor() const { return true; }
juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor() { return new AudioPluginAudioProcessorEditor (*this); }
void AudioPluginAudioProcessor::getStateInformation (juce::MemoryBlock& destData) {}
void AudioPluginAudioProcessor::setStateInformation (const void* data, int sizeInBytes) {}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new AudioPluginAudioProcessor(); }
