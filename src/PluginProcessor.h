#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <torch/torch.h>
#include <regex>
#include "Autoencoder.h"
#include "GrooveNN.h"

//==============================================================================
class AudioPluginAudioProcessor : public juce::AudioProcessor, public juce::Thread
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

    void run() override;
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
    double getBackgroundProgress() {return  backgroundProgress;}
    using PendingNote = std::tuple<int, int, int64_t>; // Channel, noteNumber, absoluteNoteOffSample
    std::vector<PendingNote> pendingNotes; // Stores pending Note Offs
    const std::array<float, 16>& getProbabilities() const { return probabilitiesArray; }
    // Slider Members
    std::atomic<float> tolerance { 0.5f };
    std::atomic<float> grooveAmount { 0.5f };

    
    // Training & Processing Functions
    void startTrainingSession(int epochs, double lr);
    void processBatch(const juce::Array<juce::File>& files);
    void processDirectory(const juce::File& dir);

    std::atomic<ThreadTask> currentTask { ThreadTask::None };
    juce::File directoryToProcess;

    // Use an atomic float to report progress back to the UI
    std::atomic<double> backgroundProgress { 0.0 };
    
    // --- Audio Analysis Functionality ---
    void loadAudioFile (const juce::File& file);
    void detectOnsets();
    void findTempoFromAudio (const juce::File& file);
    void segmentAudioFile();

    // Accessors for UI or Training
    int getDetectedBpm() const { return detectedBpm; }
    const std::vector<std::vector<int>>& getDetectedPatterns() const { return trainingPatterns; }

    // NN Functions & Variables
    torch::Tensor trainingTensor;
    int trainingEpochs = 200;
    double trainingLR = 0.001;
    void prepareTrainingTensors();
    void saveDataset(const juce::File& outputFile);
    void generateNewRhythm();
    void updateGrooveForNextBar();

private:
    // Audio and Timing member variables
    double sampleRate = 44100.0;
    double currentPosition;
    double dawBpm = 120.0;
    int64_t blockStartSample = 0;
    // int intStep = 0;
    std::atomic<double> currentStep;
    // std::atomic<int> lastStep { -1 };
    double noteLengthSeconds = 0.2;
    juce::MidiBuffer midiBuffer;
    int numSamples;
    std::atomic<int64_t> lastIteration { -1 };

    // Sequencer arrays
    std::array<int, 16> rhythmArray, pitchArray;
    std::array<float, 16> probabilitiesArray; // Stores raw model outputs

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
    std::array<float, 16> grooveMeans;   // mu
    std::array<float, 16> grooveSigmas;
    std::array<float, 16> currentGrooveShifts;
    
    // MIDI Helper Methods
//    void makeMIDINote(int noteNumber, int sampleOffset, juce::MidiBuffer& targetBuffer);
    void makeMIDINote(int noteNumber, int sampleOffset, juce::uint8 velocity);
    void processPendingNotes(juce::MidiBuffer& midiBuffer, int64_t blockStartSample, int numSamples);
    
    // Audio Analysis Members
    juce::AudioFormatManager formatManager;
    juce::AudioSampleBuffer loadedBuffer;
    std::vector<double> onsets;
    std::vector<int> steps;     // Flattened binary array
    std::vector<float> shifts;
    std::vector<std::vector<int>> trainingPatterns; // List of 16-step bars

    int detectedBpm = 120;
    int barLengthInSamples = 0;
    int sixteenthNoteSamples = 0;
    int numBars = 0;
    int numSixteenths = 0;

    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessor)
};
