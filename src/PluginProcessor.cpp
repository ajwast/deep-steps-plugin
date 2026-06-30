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
                       ),
       apvts(*this, nullptr, "Parameters", createParameterLayout()),
       trainer(analyzer, model, grooveModel)
{
    // Get the pointers now that apvts is constructed
    toleranceParameter = apvts.getRawParameterValue("tolerance");
    grooveParameter    = apvts.getRawParameterValue("grooveAmount");
    noteLengthSeconds = apvts.getRawParameterValue("noteLength");
    for (int i = 0; i < 16; ++i)
    {
        pitchParameters[i] = apvts.getRawParameterValue("pitch" + juce::String(i));
        probabilitiesArray[i].store(0.0f);
        rhythmArray[i] = 0;
        grooveMeans[i].store(0.0f);
        grooveSigmas[i].store(0.0f);
        currentGrooveShifts[i].store(0.0f);
    }

    for (int i = 0; i < 4; ++i)
    {
        latentParameters[i] = apvts.getRawParameterValue("latent" + juce::String(i));
    }
}

AudioPluginAudioProcessor::~AudioPluginAudioProcessor()
{
}

//==============================================================================
// MIDI HELPER FUNCTIONS
//==============================================================================

void AudioPluginAudioProcessor::makeMIDINote(int noteNumber, int sampleOffset, juce::uint8 velocity)
{
    auto noteOn = juce::MidiMessage::noteOn(1, noteNumber, velocity);
    midiBuffer.addEvent(noteOn, sampleOffset);


    int64_t noteOffSample = blockStartSample + sampleOffset + static_cast<int64_t>(sampleRate * noteLengthSeconds->load());

    // Lock when modifying the vector
    {
        juce::ScopedLock lock(pendingNotesLock);
        pendingNotes.push_back({noteOn.getChannel(), noteOn.getNoteNumber(), noteOffSample});
    }
}

void AudioPluginAudioProcessor::processPendingNotes(juce::MidiBuffer& targetMidiBuffer, int64_t currentBlockStart, int numSamplesInBlock)
{
    // Lock when accessing/modifying the vector
    {
        juce::ScopedLock lock(pendingNotesLock);
        for (auto it = pendingNotes.begin(); it != pendingNotes.end();)
        {
            const auto& [channel, noteNumber, noteOffSample] = *it;

            if (noteOffSample >= currentBlockStart && noteOffSample < currentBlockStart + numSamplesInBlock)
            {
                // Note-off falls within this block: emit at the correct sample offset
                int sampleOffset = static_cast<int>(noteOffSample - currentBlockStart);
                auto noteOff = juce::MidiMessage::noteOff(channel, noteNumber);
                targetMidiBuffer.addEvent(noteOff, sampleOffset);
                it = pendingNotes.erase(it);
            }
            else if (noteOffSample < currentBlockStart)
            {
                // Note-off is overdue (e.g. after a DAW scrub or skipped block):
                // emit at the start of this block to prevent a stuck note.
                auto noteOff = juce::MidiMessage::noteOff(channel, noteNumber);
                targetMidiBuffer.addEvent(noteOff, 0);
                it = pendingNotes.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }
}

//==============================================================================
void AudioPluginAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    this->sampleRate = sampleRate;
    midiBuffer.clear();

    // Lock when clearing the vector
    {
        juce::ScopedLock lock(pendingNotesLock);
        pendingNotes.clear();
    }

    blockStartSample = 0;
    lastIteration.store(-1); // Reset iteration tracking for clean transport start
    stepTriggeredInBar.fill(false); // Reset step trigger flags

    // Initialize EVERYTHING to prevent garbage logic
    rhythmArray.fill(0);
    pitchArray.fill(60);
    for (int i = 0; i < 16; ++i) {
        probabilitiesArray[i].store(0.0f);
        currentGrooveShifts[i].store(0.0f);
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

    // Get Timing Context
    dawBpm = optPos->getBpm().orFallback(120.0);
    blockStartSample = optPos->getTimeInSamples().orFallback(0);
    int numSamples = buffer.getNumSamples();
    int64_t blockEnd = blockStartSample + numSamples;

    double samplesPer16th = (getSampleRate() * 60.0) / (dawBpm * 4.0);
    double samplesPerLoop = samplesPer16th * 16.0;

    // Update UI Progress smoothly based on grid, not shifted time
    currentStep.store((double)blockStartSample / samplesPer16th);


    float scale = grooveParameter->load();
    float currentTolerance = toleranceParameter->load();
    double halfWindow = samplesPer16th * 0.5;

    // Check once per block whether we've crossed into a new bar iteration,
    //    and if so regenerate the groove shifts for all 16 steps.
    int64_t currentIteration = static_cast<int64_t>(std::floor((double)blockStartSample / samplesPerLoop));
    if (currentIteration != lastIteration.load())
    {
        updateGrooveForNextBar();
        lastIteration.store(currentIteration);
        stepTriggeredInBar.fill(false); // Reset step trigger flags for new bar
    }

    // Iterate through the 16 steps
    for (int i = 0; i < 16; ++i)
    {
        float prob = probabilitiesArray[i].load();

        // Only process if the note is likely to play
        if (prob > currentTolerance)
        {
            // Check current iteration, and one ahead/behind to catch notes
            // shifted across loop boundaries (e.g. a negative groove shift from the start of the next bar)
            for (int64_t iteration = currentIteration - 1; iteration <= currentIteration + 1; ++iteration)
            {
                // Skip if we've already triggered this step in the current bar
                // (prevents duplicate triggers when groove shifts cause overlap)
                if (stepTriggeredInBar[i])
                    break;
                
                double stepNominalSample = (iteration * samplesPerLoop) + (i * samplesPer16th);
                double nudge = currentGrooveShifts[i].load() * scale * halfWindow;
                int64_t targetSample = static_cast<int64_t>(std::round(stepNominalSample + nudge));

                // TRIGGER: Does this shifted note belong in THIS block?
                if (targetSample >= blockStartSample && targetSample < blockEnd)
                {
                    int offset = static_cast<int>(targetSample - blockStartSample);
                    // Ensure offset is strictly within buffer range for safety
                    offset = juce::jlimit(0, numSamples - 1, offset);

                    uint8_t velocity = static_cast<uint8_t>(juce::jlimit(0.0f, 1.0f, prob) * 127.0f);

                    int currentPitch = static_cast<int>(pitchParameters[i]->load());
                    makeMIDINote(currentPitch, offset, velocity);
                    
                    // Mark this step as triggered in the current bar
                    stepTriggeredInBar[i] = true;
                }
            }
        }
    }

    // Finalize MIDI
    midiMessages.addEvents(midiBuffer, 0, numSamples, 0);
    midiBuffer.clear();


    processPendingNotes(midiMessages, blockStartSample, numSamples);
}

void AudioPluginAudioProcessor::generateNewRhythm()
{
    // Generate Latent Vector
    torch::Tensor latent;

    if (isDensityEstimated()) {
        // Sample using the surveyed Mean and StdDev
        latent = getLatentMeans() + (torch::randn({1, 4}) * getLatentStdDevs());
    } else {
        // Fallback if training hasn't happened yet
        latent = (torch::rand({1, 4}) * 2.0) - 1.0;
    }

    // Update APVTS parameters - this will trigger UI and can be automated
    auto latentData = latent.data_ptr<float>();
    for (int i = 0; i < 4; ++i)
    {
        auto* param = dynamic_cast<juce::AudioParameterFloat*>(apvts.getParameter("latent" + juce::String(i)));
        if (param)
            param->setValueNotifyingHost(param->getNormalisableRange().convertTo0to1(latentData[i]));
    }

    // Run inference
    updateModelFromLatent();
}

void AudioPluginAudioProcessor::updateModelFromLatent()
{
    model.eval();
    grooveModel.eval();

    // Get latent vector from APVTS
    float latentVals[4];
    for (int i = 0; i < 4; ++i)
        latentVals[i] = latentParameters[i]->load();

    torch::Tensor latent = torch::from_blob(latentVals, {1, 4}, torch::kFloat32).clone();

    // Decode Rhythm
    auto output = model.decode(latent).view({16});
    auto outputData = output.data_ptr<float>();

    // Decode Groove
    auto rhythmTensor = output.unsqueeze(0); // Shape [1, 16]
    auto [mu, sigma] = grooveModel.forward(rhythmTensor);
    auto muData = mu.data_ptr<float>();
    auto sigmaData = sigma.data_ptr<float>();

    // Update the internal arrays (atomics)
    for (int i = 0; i < 16; ++i) {
        probabilitiesArray[i].store(outputData[i]);
        rhythmArray[i] = (outputData[i] > 0.5) ? 1 : 0;
        grooveMeans[i].store(muData[i]);
        grooveSigmas[i].store(sigmaData[i]);
    }

    updateGrooveForNextBar();
}

void AudioPluginAudioProcessor::updateGrooveForNextBar()
{

    float humanize = grooveParameter->load();
    juce::Random& r = juce::Random::getSystemRandom();

    for (int i = 0; i < 16; ++i)
    {
        // Clamp sigma to [0, 1] before use so that large model outputs cannot
        // push the shift outside the ±halfWindow region, causing missed triggers.
        float sigma = juce::jlimit(0.0f, 1.0f, grooveSigmas[i].load());
        float noise = (r.nextFloat() * 2.0f - 1.0f) + (r.nextFloat() * 2.0f - 1.0f);
        float sampled = grooveMeans[i].load() + (noise * sigma * humanize);
        currentGrooveShifts[i].store(juce::jlimit(-1.0f, 1.0f, sampled));
    }
}

void AudioPluginAudioProcessor::triggerBatchAnalysis(const juce::File& directory)
{
    trainer.triggerBatchAnalysis(directory);
}

void AudioPluginAudioProcessor::startTrainingSession(int epochs, double lr)
{
    trainer.startTrainingSession(epochs, lr);
}

double AudioPluginAudioProcessor::getBackgroundProgress()
{
    return trainer.getProgress();
}

torch::Tensor AudioPluginAudioProcessor::getLatentMeans() const
{
    return trainer.getLatentMeans();
}

torch::Tensor AudioPluginAudioProcessor::getLatentStdDevs() const
{
    return trainer.getLatentStdDevs();
}

bool AudioPluginAudioProcessor::isDensityEstimated() const
{
    return trainer.isDensityEstimated();
}

bool AudioPluginAudioProcessor::hasFinishedTraining() const
{
    return trainer.hasFinishedTraining();
}



void AudioPluginAudioProcessor::saveDataset(const juce::File& outputFile)
{
    trainer.saveDataset(outputFile);
}

void AudioPluginAudioProcessor::loadDataset(const juce::File& inputFile)
{
    trainer.loadDataset(inputFile);
}

juce::AudioProcessorValueTreeState::ParameterLayout AudioPluginAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Tolerance parameter
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID {"tolerance", 1},    // Use ParameterID object to set version hint
        "Tolerance",
        0.0f, 1.0f, 0.5f));
    // Groove
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID {"grooveAmount", 1}, // Use ParameterID object to set version hint
        "Groove Amount",
        0.0f, 1.0f, 0.5f));

    // Note Length (in seconds)
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID {"noteLength", 1},
        "Note Length",
        juce::NormalisableRange<float>(0.01f, 2.0f, 0.01f),
        0.1f));

    // Latent space parameters (4D)
    for (int i = 0; i < 4; ++i)
    {
        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID {"latent" + juce::String(i), 1},
            "Latent " + juce::String(i),
            -3.0f, 3.0f, 0.0f));
    }

    // Pitch slider params
    for (int i = 0; i < 16; ++i)
    {
        juce::String id = "pitch" + juce::String(i);
        juce::String name = "Step " + juce::String(i + 1) + " Pitch";

        params.push_back(std::make_unique<juce::AudioParameterInt>(
            juce::ParameterID {id, 1}, name, 0, 127, 60));
    }

    return { params.begin(), params.end() };
}

void AudioPluginAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void AudioPluginAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
    if (xmlState != nullptr && xmlState->hasTagName (apvts.state.getType()))
        apvts.replaceState (juce::ValueTree::fromXml (*xmlState));
}


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

void AudioPluginAudioProcessor::releaseResources()
{
    // Lock when clearing in destructor/release
    juce::ScopedLock lock(pendingNotesLock);
    pendingNotes.clear();
}

bool AudioPluginAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;
    return true;
}

bool AudioPluginAudioProcessor::hasEditor() const { return true; }
juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor() { return new AudioPluginAudioProcessorEditor (*this); }


juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new AudioPluginAudioProcessor(); }




