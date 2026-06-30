#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <torch/torch.h>
#include <regex>
#include "Autoencoder.h"
#include "GrooveNN.h"
#include "AudioAnalyzer.h"
#include "ModelTrainerThread.h"

//==============================================================================
class AudioPluginAudioProcessor : public juce::AudioProcessor
{
public:
    //==============================================================================
    AudioPluginAudioProcessor();
    ~AudioPluginAudioProcessor() override;

    enum class ThreadTask {
        None,
        Training,
        BatchAnalysis
    };

    //void run() override;
    void triggerBatchAnalysis(const juce::File& directory);
    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;
    using AudioProcessor::processBlock;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // Sequencer Timing & UI getters
    std::atomic<double>& getCurrentStep() { return currentStep; }
    const std::array<int, 16>& getRhythmArray() const { return rhythmArray; }
    std::array<int, 16>& getPitchArray() { return pitchArray; }
    double getBackgroundProgress();
    using PendingNote = std::tuple<int, int, int64_t>; // Channel, noteNumber, absoluteNoteOffSample
    std::vector<PendingNote> pendingNotes; // Stores pending Note Offs
    mutable juce::CriticalSection pendingNotesLock;  // Mutex to protect pendingNotes vector
    const std::array<std::atomic<float>, 16>& getProbabilities() const { return probabilitiesArray; }

    // Latent space stats for UI
    torch::Tensor getLatentMeans() const;
    torch::Tensor getLatentStdDevs() const;
    bool isDensityEstimated() const;
    bool hasFinishedTraining() const;

    // Value Tree State & Parameters
    juce::AudioProcessorValueTreeState apvts;
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    std::atomic<float>* noteLengthSeconds = nullptr;
    std::atomic<float>* toleranceParameter = nullptr;
    std::atomic<float>* grooveParameter = nullptr;
    std::array<std::atomic<float>*, 16> pitchParameters;
    std::array<std::atomic<float>*, 4> latentParameters;

    // Training & Processing Functions (delegated to trainer)
    void startTrainingSession(int epochs, double lr);
    // void triggerBatchAnalysis(const juce::File& directory);
    void saveDataset(const juce::File& outputFile);
    void loadDataset(const juce::File& inputFile);
    //
    void generateNewRhythm();
    void updateGrooveForNextBar();
    void updateModelFromLatent();

private:
    // Audio and Timing member variables
    double sampleRate = 44100.0;
    double currentPosition;
    double dawBpm = 120.0;
    int64_t blockStartSample = 0;
    // int intStep = 0;
    std::atomic<double> currentStep;
    // std::atomic<int> lastStep { -1 };
    // double noteLengthSeconds = 0.2;
    juce::MidiBuffer midiBuffer;
    int numSamples;
    std::atomic<int64_t> lastIteration { -1 };
    
    // Track which steps have been triggered in the current bar to prevent duplicates
    std::array<bool, 16> stepTriggeredInBar; // Track per-step triggers within a bar

    // Sequencer arrays
    std::array<int, 16> rhythmArray, pitchArray;
    std::array<std::atomic<float>, 16> probabilitiesArray; // Stores raw model outputs

    // Parallel containers for rhythm (binary) and groove (offsets)
    std::vector<std::vector<float>> masterRhythmDataset;
    std::vector<std::vector<float>> masterGrooveDataset;

    // Separate tensors for training the two models
    torch::Tensor trainingRhythmTensor;
    torch::Tensor trainingGrooveTensor;
    
    std::vector<float> trainingData;
    std::atomic<bool> finishedTraining { false };

    // Neural Network
    Autoencoder model;

    GaussianGrooveModel grooveModel;
    std::unique_ptr<torch::optim::Adam> grooveOptimizer;
    std::array<std::atomic<float>, 16> grooveMeans;   // mu
    std::array<std::atomic<float>, 16> grooveSigmas;
    std::array<std::atomic<float>, 16> currentGrooveShifts;

    // Audio Analysis & background training thread
    AudioAnalyzer analyzer;
    ModelTrainerThread trainer;

    // MIDI Helper Methods
    void makeMIDINote(int noteNumber, int sampleOffset, juce::uint8 velocity);
    void processPendingNotes(juce::MidiBuffer& midiBuffer, int64_t blockStartSample, int numSamples);

    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessor)
};
