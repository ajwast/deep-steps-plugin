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
    formatManager.registerBasicFormats();
}

AudioPluginAudioProcessor::~AudioPluginAudioProcessor() {
    stopThread(2000);
}

//==============================================================================
// MIDI HELPER FUNCTIONS
//==============================================================================

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

    dawBpm = optPos->getBpm().orFallback(120.0);
    blockStartSample = optPos->getTimeInSamples().orFallback(0);
    
    numSamples = buffer.getNumSamples();

    // 1. Calculate timing once per block instead of inside the loop if possible
    // But for sample-accurate MIDI triggering, we check samples:
    double samplesPer16th = (sampleRate * 60.0) / (dawBpm * 4.0);

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

void AudioPluginAudioProcessor::loadAudioFile (const juce::File& file)
{
    if (auto* reader = formatManager.createReaderFor(file))
    {
        currentSampleRate = reader->sampleRate;
        loadedBuffer.setSize((int)reader->numChannels, (int)reader->lengthInSamples);
        reader->read(&loadedBuffer, 0, (int)reader->lengthInSamples, 0, true, true);
        delete reader;

        detectOnsets();
        findTempoFromAudio(file);
        segmentAudioFile();

        // Print results to console as requested
        std::cout << "--- Analysis Results ---" << std::endl;
        std::cout << "Detected BPM: " << detectedBpm << std::endl;
        std::cout << "Bars: " << numBars + 1 << std::endl;
        std::cout << steps << std::endl;
        std::cout << shifts << std::endl;
 //        std::cout << "Patterns Extracted: " << trainingPatterns.size() << std::endl;
//
//        for (size_t i = 0; i < trainingPatterns.size(); ++i) {
//            std::cout << "Bar " << i << ": ";
//            for (int step : trainingPatterns[i]) std::cout << step;
//            std::cout << std::endl;
//        }
    }
}

void AudioPluginAudioProcessor::findTempoFromAudio (const juce::File& file)
{
    detectedBpm = 0; // reset

    const juce::String fileName = file.getFileNameWithoutExtension();
    std::string nameStd = fileName.toStdString();

    //Try "[number] bpm" with optional underscores
    std::regex bpmRegex ("(\\d{2,3})[_\\s]*[Bb][Pp][Mm]");
    std::smatch match;

    if (std::regex_search(nameStd, match, bpmRegex))
    {
        int value = std::stoi(match[1].str());
        if (value >= 80 && value <= 170)
        {
            detectedBpm = value;
            return;
        }
    }

    // Else: any number between 80–170
    std::regex numRegex ("(?:^|[^\\d])([8-9][0-9]|1[0-6][0-9]|170)(?:$|[^\\d])");
    auto begin = std::sregex_iterator(nameStd.begin(), nameStd.end(), numRegex);
    auto end   = std::sregex_iterator();

    for (auto it = begin; it != end; ++it)
    {
        int value = std::stoi((*it)[1].str());
        if (value >= 80 && value <= 170)
        {
            detectedBpm = value;
            return;
        }
    }
}

void AudioPluginAudioProcessor::detectOnsets()
{
    onsets.clear();

    const int numSamples  = loadedBuffer.getNumSamples();
    const int numChannels = loadedBuffer.getNumChannels();
    if (numSamples == 0 || numChannels == 0)
        return;

    // === Parameters ===
    constexpr int fftOrder   = 10;                  // 2^10 = 1024
    constexpr int fftSize    = 1 << fftOrder;       // 1024-point FFT
    constexpr int hopSize    = fftSize / 2;         // 50% overlap
    constexpr float peakThreshold = 0.15f;          // onset sensitivity
    constexpr int minGapSamples = 2048 * 2;             // ~50 ms at 44.1kHz

    juce::dsp::FFT fft (fftOrder);
    juce::dsp::WindowingFunction<float> window (fftSize, juce::dsp::WindowingFunction<float>::hann);

    std::vector<float> fluxValues;
    std::vector<float> prevMag (fftSize / 2, 0.0f);

    // Temporary buffers
    std::vector<float> fftInput (fftSize, 0.0f);
    std::vector<std::complex<float>> fftOutput (fftSize);

    // === Process frames ===
    for (int start = 0; start + fftSize < numSamples; start += hopSize)
    {
        // Mix down to mono
        for (int i = 0; i < fftSize; ++i)
        {
            float sum = 0.0f;
            for (int ch = 0; ch < numChannels; ++ch)
                sum += loadedBuffer.getSample (ch, start + i);
            fftInput[i] = sum / (float) numChannels;
        }

        // Apply window
        window.multiplyWithWindowingTable (fftInput.data(), fftSize);

        // FFT needs real + imag interleaved in a float array
        std::vector<float> fftData (2 * fftSize, 0.0f);
        for (int i = 0; i < fftSize; ++i)
        {
            fftData[2 * i]     = fftInput[i];
            fftData[2 * i + 1] = 0.0f;
        }

        // Run FFT
        fft.performRealOnlyForwardTransform (fftData.data());

        // Magnitude spectrum
        std::vector<float> mag (fftSize / 2, 0.0f);
        for (int bin = 0; bin < fftSize / 2; ++bin)
        {
            float re = fftData[2 * bin];
            float im = fftData[2 * bin + 1];
            mag[bin] = std::sqrt (re * re + im * im);
        }

        // Spectral flux: sum of positive differences
        float flux = 0.0f;
        for (int bin = 0; bin < fftSize / 2; ++bin)
        {
            float diff = mag[bin] - prevMag[bin];
            if (diff > 0.0f)
                flux += diff;
        }
        fluxValues.push_back (flux);

        prevMag = mag;
    }

    if (fluxValues.empty())
        return;

    // === Normalize flux values ===
    float maxFlux = *std::max_element (fluxValues.begin(), fluxValues.end());
    if (maxFlux > 0.0f)
        for (auto& f : fluxValues)
            f /= maxFlux;

    // === Peak picking with simple threshold ===
    int lastOnsetSample = -minGapSamples;
    for (size_t i = 1; i < fluxValues.size() - 1; ++i)
    {
        float val = fluxValues[i];
        if (val > peakThreshold && val > fluxValues[i - 1] && val > fluxValues[i + 1])
        {
            int sampleIndex = (int) (i * hopSize);
            if (sampleIndex - lastOnsetSample >= minGapSamples)
            {
                double timeSec = sampleIndex / sampleRate;
                onsets.push_back (timeSec);
                lastOnsetSample = sampleIndex;
            }
        }
    }
}

void AudioPluginAudioProcessor::segmentAudioFile()
{
    const int numSamples  = loadedBuffer.getNumSamples();
    if (numSamples == 0 || detectedBpm <= 0 || sampleRate <= 0.0)
        return;

    // Length of one quarter note (beat) in samples
    double quarterNoteSamples = (60.0 / detectedBpm) * sampleRate;

    // One bar (4/4 assumption for now)
    barLengthInSamples = (int)std::round(quarterNoteSamples * 4.0);

    // One sixteenth note = quarter / 4
    sixteenthNoteSamples = (int)std::round(quarterNoteSamples / 4.0);

    // How many bars & 16ths fit in the file
    numBars = (int)std::floor((double)numSamples / barLengthInSamples);
    numSixteenths = (int)std::floor((double)numSamples / sixteenthNoteSamples);

    // prepare/clear step + shift arrays
    steps.assign(numSixteenths, 0);
    shifts.assign(numSixteenths, 0.0f);

    if (onsets.empty())
        return;

    // half of a 32nd note in samples = half of a 16th
    const double half32ndSamples = (double)sixteenthNoteSamples * 0.5;

    for (double onsetSec : onsets)
    {
        // convert onset time (seconds) to sample
        double onsetSample = onsetSec * sampleRate;

        // find nearest 16th index
        int nearestIdx = (int) std::lrint(onsetSample / (double)sixteenthNoteSamples);
        if (nearestIdx < 0 || nearestIdx >= numSixteenths)
            continue;

        // center sample of that 16th
        double centerSample = (double)nearestIdx * (double)sixteenthNoteSamples;

        // allowed snapping window = one 32nd before and after
        double windowStart = centerSample - half32ndSamples;
        double windowEnd   = centerSample + half32ndSamples;

        if (onsetSample >= windowStart && onsetSample < windowEnd)
        {
            steps[nearestIdx] = 1;

            // deviation from grid (in samples)
            double offset = onsetSample - centerSample;

            // normalize: -1.0 = late by half a 16th, +1.0 = early by half a 16th
            float normalized = juce::jlimit(-1.0, 1.0, offset / half32ndSamples);

            shifts[nearestIdx] = normalized;
        }
    }
    

    // Convert into 16-step segments (Training Data)
    for (int bar = 0; bar < numSixteenths / 16; ++bar) {
        std::vector<int> pattern;
        for (int step = 0; step < 16; ++step) {
            pattern.push_back(steps[bar * 16 + step]);
        }
        trainingPatterns.push_back(pattern);
    }
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
