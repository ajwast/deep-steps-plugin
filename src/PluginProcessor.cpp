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

void AudioPluginAudioProcessor::makeMIDINote(int noteNumber, int sampleOffset, juce::uint8 velocity)
{
    auto noteOn = juce::MidiMessage::noteOn(1, noteNumber, velocity);
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
    blockStartSample = 0;

    // Initialize EVERYTHING to prevent garbage logic
    rhythmArray.fill(0);
    pitchArray.fill(60);
    probabilitiesArray.fill(0.0f);
    currentGrooveShifts.fill(0.0f);

    pendingNotes.clear();

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

}

void AudioPluginAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    for (auto i = getTotalNumInputChannels(); i < getTotalNumOutputChannels(); ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    auto* playHead = getPlayHead();
    if (playHead == nullptr) return;
    auto optPos = playHead->getPosition();
    if (!optPos.hasValue()) return;

    // 1. Get Timing Context
    dawBpm = optPos->getBpm().orFallback(120.0);
    blockStartSample = optPos->getTimeInSamples().orFallback(0);
    int numSamples = buffer.getNumSamples();
    int64_t blockEnd = blockStartSample + numSamples;

    double samplesPer16th = (getSampleRate() * 60.0) / (dawBpm * 4.0);
    double samplesPerLoop = samplesPer16th * 16.0;

    // Update UI Progress smoothly based on grid, not shifted time
    currentStep.store((double)blockStartSample / samplesPer16th);


    float scale = grooveAmount.load();
    float currentTolerance = tolerance.load();
    double halfWindow = samplesPer16th * 0.5;

    // 2. Iterate through the 16 steps
    for (int i = 0; i < 16; ++i)
    {
        // Only process if the note is likely to play
        if (probabilitiesArray[i] > currentTolerance)
        {
            // Determine which "iteration" of the 16-step loop we are currently in
            int64_t currentIteration = static_cast<int64_t>(std::floor((double)blockStartSample / samplesPerLoop));

            // Update groove if needed
            if (currentIteration != lastIteration.load())
            {
                updateGrooveForNextBar();
                lastIteration.store(currentIteration);
            }
            // Check current iteration, and one ahead/behind to catch notes
            // shifted across loop boundaries (e.g., negative shift from the start of the next bar)
            for (int64_t iteration = currentIteration - 1; iteration <= currentIteration + 1; ++iteration)
            {
                double stepNominalSample = (iteration * samplesPerLoop) + (i * samplesPer16th);
                double nudge = currentGrooveShifts[i] * scale * halfWindow;
                int64_t targetSample = static_cast<int64_t>(std::round(stepNominalSample + nudge));

                // TRIGGER: Does this shifted note belong in THIS block?
                if (targetSample >= blockStartSample && targetSample < blockEnd)
                {
                    int offset = static_cast<int>(targetSample - blockStartSample);
                    // Ensure offset is strictly within buffer range for safety
                    offset = juce::jlimit(0, numSamples - 1, offset);

                    uint8_t velocity = static_cast<uint8_t>(juce::jlimit(0.0f, 1.0f, probabilitiesArray[i]) * 127.0f);

                    makeMIDINote(pitchArray[i], offset, velocity);
                }
            }
        }
    }

    // 3. Finalize MIDI
    midiMessages.addEvents(midiBuffer, 0, numSamples, 0);
    midiBuffer.clear();
    processPendingNotes(midiMessages, blockStartSample, numSamples);
}

void AudioPluginAudioProcessor::generateNewRhythm()
{
    model.eval();
    grooveModel.eval();

    // 1. Generate Rhythm (Current Logic)
    auto input = torch::randn({1, 4});
    auto output = model.decode(input).view({16});

    // 2. Generate Groove (Sampling from the Gaussian)
    auto rhythmTensor = output.unsqueeze(0); // Shape [1, 16]
    auto [mu, sigma] = grooveModel.forward(rhythmTensor);

    auto muData = mu.data_ptr<float>();
    auto sigmaData = sigma.data_ptr<float>();

    // 3. Update the internal arrays for the Editor and processBlock
    auto outputData = output.data_ptr<float>();
    // auto shiftData = sampledShifts.data_ptr<float>();

    for (int i = 0; i < 16; ++i) {
        probabilitiesArray[i] = outputData[i];
        rhythmArray[i] = (outputData[i] > 0.5) ? 1 : 0;
        grooveMeans[i] = muData[i];
        grooveSigmas[i] = sigmaData[i];

    }

    updateGrooveForNextBar();

}

void AudioPluginAudioProcessor::updateGrooveForNextBar()
{
    float humanize = tolerance.load();
    juce::Random& r = juce::Random::getSystemRandom();

    for (int i = 0; i < 16; ++i)
    {
        // Simple Box-Muller or basic random approximation of the Gaussian
        float noise = (r.nextFloat() * 2.0f - 1.0f) + (r.nextFloat() * 2.0f - 1.0f);
        float sampled = grooveMeans[i] + (noise * grooveSigmas[i] * humanize);
        currentGrooveShifts[i] = juce::jlimit(-1.0f, 1.0f, sampled);
    }

}

void AudioPluginAudioProcessor::triggerBatchAnalysis(const juce::File& directory)
{
    if (!isThreadRunning())
    {
        directoryToProcess = directory;
        currentTask = ThreadTask::BatchAnalysis;
        backgroundProgress = 0.0f;
        startThread();
    }
}

void AudioPluginAudioProcessor::startTrainingSession(int epochs, double lr)
{
    if (!isThreadRunning()) {
        currentTask = ThreadTask::Training;
        backgroundProgress = 0.0f;

        // Check if we have data prepared from the batch processor
        if (trainingRhythmTensor.numel() > 0 && !isThreadRunning())
        {
            trainingEpochs = epochs;
            trainingLR = lr;

            juce::Logger::writeToLog("Starting training thread with " +
                                    juce::String(trainingRhythmTensor.size(0)) + " patterns...");
            startThread();
        }
        else if (isThreadRunning()) {
            juce::Logger::writeToLog("Training is already in progress.");
        }
        else {
            juce::Logger::writeToLog("Training failed to start: No data in trainingRhythmTensor.");
        }

    }

}

void AudioPluginAudioProcessor::run()
{

    while (!threadShouldExit())
    {
        if (currentTask == ThreadTask::BatchAnalysis)
        {
            // Perform the heavy lifting here
            processDirectory(directoryToProcess);

            currentTask = ThreadTask::None;
            backgroundProgress = 1.0f; // Done
            return; // Exit thread when done
        }
        else if (currentTask == ThreadTask::Training)
        {
            // Safety check
            if (trainingRhythmTensor.numel() == 0 || trainingGrooveTensor.numel() == 0) {
                std::cout << "[Error] run() called but tensors are empty!" << std::endl;
                return;
            }

            std::cout << "--- Training Started ---" << std::endl;

            // Initialize optimizers with the learning rate passed from the UI
            torch::optim::Adam rhythmOptimizer(model.parameters(), torch::optim::AdamOptions(trainingLR));
            torch::optim::Adam grooveOptimizer(grooveModel.parameters(), torch::optim::AdamOptions(trainingLR));

            model.train();
            grooveModel.train();

            for (int epoch = 0; epoch < trainingEpochs; ++epoch)
            {
                if (threadShouldExit()) {
                    std::cout << "Training aborted by user." << std::endl;
                    return;
                }

                // 1. Train Rhythm Autoencoder
                rhythmOptimizer.zero_grad();
                auto rhythmPred = model.forward(trainingRhythmTensor);
                auto rhythmLoss = torch::nn::functional::binary_cross_entropy(rhythmPred, trainingRhythmTensor);
                rhythmLoss.backward();
                rhythmOptimizer.step();

                // 2. Train Gaussian Groove Model (Masked)
                grooveOptimizer.zero_grad();
                auto [mu, sigma] = grooveModel.forward(trainingRhythmTensor);
                auto grooveLoss = calculateGaussianLoss(mu, sigma, trainingGrooveTensor, trainingRhythmTensor);
                grooveLoss.backward();
                grooveOptimizer.step();

                // 3. Log progress every 10 epochs
                if (epoch % 10 == 0 || epoch == trainingEpochs - 1)
                {
                    std::cout << "Epoch: " << epoch
                              << " | Rhythm Loss: " << rhythmLoss.item<float>()
                              << " | Groove Loss: " << grooveLoss.item<float>() << std::endl;
                }
            }

            finishedTraining = true;
            std::cout << "--- Training Finished Successfully ---" << std::endl;

            currentTask = ThreadTask::None;
            return;
        }

        // Avoid burning CPU if there's nothing to do
        wait(100);
    }
}

void AudioPluginAudioProcessor::processDirectory(const juce::File& dir)
{
    auto files = dir.findChildFiles(juce::File::findFiles, false, "*.wav");
    int totalFiles = files.size();

    for (int i = 0; i < totalFiles; ++i)
    {
        if (threadShouldExit()) return;

        // Perform analysis
        // We pass the file and collect results without touching member buffers yet
        loadAudioFile(files[i]);

        // Only process if loadAudioFile actually produced steps
        if (steps.size() >= 16)
        {
            int totalBars = (int)steps.size() / 16;
            for (int bar = 0; bar < totalBars; ++bar)
            {
                std::vector<float> rhythmBar;
                std::vector<float> grooveBar;

                for (int j = 0; j < 16; ++j) {
                    int index = (bar * 16) + j;
                    rhythmBar.push_back((float)steps[index]);
                    grooveBar.push_back(shifts[index]);
                }

                masterRhythmDataset.push_back(rhythmBar);
                masterGrooveDataset.push_back(grooveBar);
            }
        }

        backgroundProgress.store((float)i / (float)totalFiles);
    }

    juce::Logger::writeToLog("Batch Complete. Bars processed: " + juce::String(masterRhythmDataset.size()));
    // Convert to tensors only after ALL files are processed
    prepareTrainingTensors();
}

void AudioPluginAudioProcessor::loadAudioFile (const juce::File& file)
{
    if (auto* reader = formatManager.createReaderFor(file))
    {
        sampleRate = reader->sampleRate;
        loadedBuffer.setSize((int)reader->numChannels, (int)reader->lengthInSamples);
        reader->read(&loadedBuffer, 0, (int)reader->lengthInSamples, 0, true, true);
        delete reader;

        detectOnsets(); // run offline analysis
        findTempoFromAudio(file);
        segmentAudioFile();


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
    if (numSamples == 0 || numChannels == 0 || sampleRate <= 0.0)
        return;

    // === Parameters ===
    constexpr int fftOrder = 10;                 // 1024
    constexpr int fftSize  = 1 << fftOrder;
    constexpr int hopSize  = fftSize / 2;
    constexpr float peakThreshold = 0.15f;
    constexpr int minGapSamples = 2048 * 2;

    juce::dsp::FFT fft (fftOrder);
    juce::dsp::WindowingFunction<float> window (
        fftSize, juce::dsp::WindowingFunction<float>::hann);

    // === Zero-padding at start ===
    const int padding = fftSize;
    const int paddedNumSamples = numSamples + padding;

    std::vector<float> monoBuffer (paddedNumSamples, 0.0f);

    for (int i = 0; i < numSamples; ++i)
    {
        float sum = 0.0f;
        for (int ch = 0; ch < numChannels; ++ch)
            sum += loadedBuffer.getSample (ch, i);

        monoBuffer[i + padding] = sum / (float) numChannels;
    }

    std::vector<float> prevMag (fftSize / 2, 0.0f);
    std::vector<float> fluxValues;
    std::vector<int>   frameToSample;

    bool firstFrame = true;

    // === Process frames ===
    for (int start = 0; start + fftSize < paddedNumSamples; start += hopSize)
    {
        std::vector<float> fftData (2 * fftSize, 0.0f);

        for (int i = 0; i < fftSize; ++i)
            fftData[2 * i] = monoBuffer[start + i];

        window.multiplyWithWindowingTable (fftData.data(), fftSize);
        fft.performRealOnlyForwardTransform (fftData.data());

        std::vector<float> mag (fftSize / 2, 0.0f);
        for (int bin = 0; bin < fftSize / 2; ++bin)
        {
            float re = fftData[2 * bin];
            float im = fftData[2 * bin + 1];
            mag[bin] = std::sqrt (re * re + im * im);
        }

        // Prime prevMag with first frame
        if (firstFrame)
        {
            prevMag = mag;
            firstFrame = false;
            continue;
        }

        float flux = 0.0f;
        for (int bin = 0; bin < fftSize / 2; ++bin)
        {
            float diff = mag[bin] - prevMag[bin];
            if (diff > 0.0f)
                flux += diff;
        }

        fluxValues.push_back (flux);
        frameToSample.push_back (start - padding);
        prevMag = mag;
    }

    if (fluxValues.empty())
        return;

    // === Normalize flux ===
    float maxFlux = *std::max_element (fluxValues.begin(), fluxValues.end());
    if (maxFlux > 0.0f)
        for (auto& f : fluxValues)
            f /= maxFlux;

    // === Peak picking (includes first frame) ===
    int lastOnsetSample = -minGapSamples;

    for (size_t i = 0; i < fluxValues.size(); ++i)
    {
        float prev = (i == 0) ? 0.0f : fluxValues[i - 1];
        float next = (i == fluxValues.size() - 1) ? 0.0f : fluxValues[i + 1];

        if (fluxValues[i] > peakThreshold &&
            fluxValues[i] > prev &&
            fluxValues[i] > next)
        {
            int sampleIndex = std::max (0, frameToSample[i]);

            if (sampleIndex - lastOnsetSample >= minGapSamples)
            {
                double timeSec = (double) sampleIndex / sampleRate;
                onsets.push_back (timeSec);
                lastOnsetSample = sampleIndex;
            }
        }
    }
}

void AudioPluginAudioProcessor::segmentAudioFile()
{
    const int numSamples = loadedBuffer.getNumSamples();
    if (numSamples == 0 ||detectedBpm <= 0 || sampleRate <= 0.0)
        return;

    // === Grid sizes ===
    const double quarterNoteSamples = (60.0 / detectedBpm) * sampleRate;
    barLengthInSamples = (int) std::round(quarterNoteSamples * 4.0);
    sixteenthNoteSamples = (int) std::round(quarterNoteSamples / 4.0);

    numBars = (int) std::ceil((double)numSamples / barLengthInSamples);
    numSixteenths = (int) std::ceil((double)numSamples / sixteenthNoteSamples);

    steps.assign(numSixteenths, 0);
    shifts.assign(numSixteenths, 0.0f);

    if (onsets.empty())
        return;

    const double half32ndSamples = sixteenthNoteSamples * 0.5;

    for (double onsetSec : onsets)
    {
        const double onsetSample = onsetSec * sampleRate;

        // nearest 16th index
        int nearestIdx = (int) std::lrint(onsetSample / sixteenthNoteSamples);

        // clamp instead of reject
        nearestIdx = juce::jlimit(0, numSixteenths - 1, nearestIdx);

        // center of this 16th
        const double centerSample = nearestIdx * sixteenthNoteSamples;

        // snapping window
        double windowStart = centerSample - half32ndSamples;
        double windowEnd   = centerSample + half32ndSamples;

        // clamp window to file bounds
        windowStart = std::max(0.0, windowStart);
        windowEnd   = std::min((double) numSamples, windowEnd);

        if (onsetSample >= windowStart && onsetSample < windowEnd)
        {
            steps[nearestIdx] = 1;
//            stepsT[nearestIdx] = 1;
            const double offset = onsetSample - centerSample;
            shifts[nearestIdx] = juce::jlimit(
                -1.0f, 1.0f,
                (float) (offset / half32ndSamples)
            );
//            shiftsT[nearestIdx] = juce::jlimit(
//                -1.0f, 1.0f,
//                (float) (offset / half32ndSamples)
//            );
        }
    }
    
}

void AudioPluginAudioProcessor::processBatch(const juce::Array<juce::File>& files)
{
    for (const auto& file : files)
    {
        if (threadShouldExit()) return;if (threadShouldExit()) return;

        if (file.isDirectory()) {
            juce::Array<juce::File> children;
            file.findChildFiles(children, juce::File::findFiles, true, "*.wav;*.aif");
            processBatch(children);
        }
        else {

            loadAudioFile(file); // This calls segmentAudioFile() internally
            
            // Slice the full-file vectors (steps/shifts) into 16-step bars
            int totalBars = (int)steps.size() / 16;

            for (int bar = 0; bar < totalBars; ++bar)
            {
                std::vector<float> rhythmBar;
                std::vector<float> grooveBar;

                for (int i = 0; i < 16; ++i) {
                    int index = (bar * 16) + i;
                    rhythmBar.push_back((float)steps[index]);
                    grooveBar.push_back(shifts[index]);
                }

                masterRhythmDataset.push_back(rhythmBar);
                masterGrooveDataset.push_back(grooveBar);
            }
        }
        // // Update progress for the UI
        // backgroundProgress.store((float)i / (float)total);
    }
    
    juce::Logger::writeToLog("Batch Complete. Bars processed: " + juce::String(masterRhythmDataset.size()));
    prepareTrainingTensors();
}

void AudioPluginAudioProcessor::prepareTrainingTensors()
{
    if (masterRhythmDataset.empty()) return;

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    int64_t numRows = (int64_t)masterRhythmDataset.size();

    // 1. Flatten and create Rhythm Tensor
    std::vector<float> flatRhythm;
    for (const auto& row : masterRhythmDataset) flatRhythm.insert(flatRhythm.end(), row.begin(), row.end());
    trainingRhythmTensor = torch::from_blob(flatRhythm.data(), {numRows, 16}, options).clone();

    // 2. Flatten and create Groove Tensor
    std::vector<float> flatGroove;
    for (const auto& row : masterGrooveDataset) flatGroove.insert(flatGroove.end(), row.begin(), row.end());
    trainingGrooveTensor = torch::from_blob(flatGroove.data(), {numRows, 16}, options).clone();
}

// void AudioPluginAudioProcessor::saveDataset(const juce::File& outputFile)
// {
//     if (trainingRhythmTensor.numel() == 0 || trainingGrooveTensor.numel() == 0)
//         return;
//
//     try {
//         // Create a map to hold both tensors
//         std::map<std::string, torch::Tensor> tensorMap;
//         tensorMap["rhythm"] = trainingRhythmTensor;
//         tensorMap["groove"] = trainingGrooveTensor;
//
//         // Save the map to a single file
//         torch::save(tensorMap, outputFile.getFullPathName().toStdString());
//
//         juce::Logger::writeToLog("Dual-Model Dataset saved successfully.");
//     }
//     catch (const std::exception& e) {
//         juce::Logger::writeToLog("Save Error: " + juce::String(e.what()));
//     }
// }



//==============================================================================
// Standard JUCE boilerplate
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
