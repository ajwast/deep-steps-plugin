#pragma once

#include <juce_core/juce_core.h>
#include <torch/torch.h>
#include "Autoencoder.h"
#include "GrooveNN.h"
#include "AudioAnalyzer.h"

class ModelTrainerThread : public juce::Thread
{
public:
    ModelTrainerThread (AudioAnalyzer& analyzer,
                        Autoencoder& model,
                        GaussianGrooveModel& grooveModel);
    ~ModelTrainerThread() override;

    enum class Task
    {
        None,
        Training,
        BatchAnalysis
    };

    void run() override;

    void triggerBatchAnalysis (const juce::File& directory);
    void startTrainingSession (int epochs, double lr);

    void saveDataset (const juce::File& file);
    void loadDataset (const juce::File& file);

    // Getters for UI
    double getProgress() const { return backgroundProgress.load(); }
    bool hasFinishedTraining() const { return finishedTraining.load(); }
    bool isDensityEstimated() const { return densityEstimated.load(); }
    torch::Tensor getLatentMeans() const { return latentMeans; }
    torch::Tensor getLatentStdDevs() const { return latentStdDevs; }

    // Tensors (for safety checking in startTraining)
    int64_t getNumPatterns() const { return trainingRhythmTensor.numel() > 0 ? trainingRhythmTensor.size(0) : 0; }

private:
    void processDirectory (const juce::File& dir);
    void prepareTrainingTensors();
    void estimateLatentDensity();

    AudioAnalyzer& analyzer;
    Autoencoder& model;
    GaussianGrooveModel& grooveModel;

    std::atomic<Task> currentTask { Task::None };
    juce::File directoryToProcess;

    int trainingEpochs = 200;
    double trainingLR = 0.001;

    // Dataset & Tensors
    std::vector<std::vector<float>> masterRhythmDataset;
    std::vector<std::vector<float>> masterGrooveDataset;
    torch::Tensor trainingRhythmTensor;
    torch::Tensor trainingGrooveTensor;

    // Latent space shape
    torch::Tensor latentMeans;
    torch::Tensor latentStdDevs;

    std::atomic<double> backgroundProgress { 0.0 };
    std::atomic<bool> finishedTraining { false };
    std::atomic<bool> densityEstimated { false };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ModelTrainerThread)
};
