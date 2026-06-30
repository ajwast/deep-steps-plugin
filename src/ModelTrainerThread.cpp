#include "ModelTrainerThread.h"
#include <iostream>

ModelTrainerThread::ModelTrainerThread (AudioAnalyzer& a, Autoencoder& m, GaussianGrooveModel& g)
    : juce::Thread ("TrainingThread"), analyzer (a), model (m), grooveModel (g)
{
}

ModelTrainerThread::~ModelTrainerThread()
{
    stopThread (2000);
}

void ModelTrainerThread::run()
{
    while (!threadShouldExit())
    {
        if (currentTask == Task::BatchAnalysis)
        {
            processDirectory (directoryToProcess);
            currentTask = Task::None;
            backgroundProgress = 1.0;
            return;
        }
        else if (currentTask == Task::Training)
        {
            if (trainingRhythmTensor.numel() == 0 || trainingGrooveTensor.numel() == 0)
            {
                std::cout << "[Error] run() called but tensors are empty!" << std::endl;
                currentTask = Task::None;
                backgroundProgress = 0.0;
                return;
            }

            std::cout << "--- Training Started (RAE Regularization Enabled) ---" << std::endl;

            auto rhythmOptions = torch::optim::AdamOptions(trainingLR).weight_decay(1e-5);
            torch::optim::Adam rhythmOptimizer(model.parameters(), rhythmOptions);
            torch::optim::Adam grooveOptimizer(grooveModel.parameters(), torch::optim::AdamOptions(trainingLR));

            model.train();
            grooveModel.train();

            const float lambda = 0.01f;

            for (int epoch = 0; epoch < trainingEpochs; ++epoch)
            {
                if (threadShouldExit()) return;

                backgroundProgress.store((double)epoch / (double)trainingEpochs);

                // Rhythm Autoencoder Training
                rhythmOptimizer.zero_grad();
                auto latent = model.encoder->forward(trainingRhythmTensor);
                auto rhythmPred = model.decoder->forward(latent);

                auto reconLoss = torch::nn::functional::binary_cross_entropy(rhythmPred, trainingRhythmTensor);
                auto latentPenalty = torch::mean(torch::pow(latent, 2));
                auto totalRhythmLoss = reconLoss + (lambda * latentPenalty);

                totalRhythmLoss.backward();
                rhythmOptimizer.step();

                // Groove Model Training
                grooveOptimizer.zero_grad();
                auto [mu, sigma] = grooveModel.forward(trainingRhythmTensor);
                auto grooveLoss = calculateGaussianLoss(mu, sigma, trainingGrooveTensor, trainingRhythmTensor);

                grooveLoss.backward();
                grooveOptimizer.step();

                if (epoch % 10 == 0 || epoch == trainingEpochs - 1)
                {
                    std::cout << "Epoch: " << epoch
                              << " | Recon: " << reconLoss.item<float>()
                              << " | Latent Pen: " << latentPenalty.item<float>()
                              << " | Groove Loss: " << grooveLoss.item<float>() << std::endl;
                }
            }

            finishedTraining = true;
            estimateLatentDensity();
            std::cout << "--- Training Finished Successfully ---" << std::endl;
            backgroundProgress.store(0.0);
            currentTask = Task::None;
            return;
        }

        wait(100);
    }
}

void ModelTrainerThread::triggerBatchAnalysis(const juce::File& directory)
{
    if (!isThreadRunning())
    {
        directoryToProcess = directory;
        currentTask = Task::BatchAnalysis;
        backgroundProgress = 0.0f;
        startThread();
    }
}

void ModelTrainerThread::startTrainingSession(int epochs, double lr)
{
    if (!isThreadRunning())
    {
        currentTask = Task::Training;
        backgroundProgress = 0.0f;
        finishedTraining = false;

        if (trainingRhythmTensor.numel() > 0)
        {
            trainingEpochs = epochs;
            trainingLR = lr;

            juce::Logger::writeToLog("Starting training thread with " +
                                    juce::String(trainingRhythmTensor.size(0)) + " patterns...");
            startThread();
        }
        else
        {
            juce::Logger::writeToLog("Training failed to start: No data in trainingRhythmTensor.");
            currentTask = Task::None;
        }
    }
}

void ModelTrainerThread::processDirectory(const juce::File& dir)
{
    auto files = dir.findChildFiles(juce::File::findFiles, false, "*.wav");
    int totalFiles = files.size();

    masterRhythmDataset.clear();
    masterGrooveDataset.clear();

    for (int i = 0; i < totalFiles; ++i)
    {
        if (threadShouldExit()) return;

        auto result = analyzer.analyzeFile(files[i]);

        if (result.steps.size() >= 16)
        {
            int totalBars = (int)result.steps.size() / 16;
            for (int bar = 0; bar < totalBars; ++bar)
            {
                std::vector<float> rhythmBar;
                std::vector<float> grooveBar;

                for (int j = 0; j < 16; ++j)
                {
                    int index = (bar * 16) + j;
                    rhythmBar.push_back((float)result.steps[index]);
                    grooveBar.push_back(result.shifts[index]);
                }

                masterRhythmDataset.push_back(rhythmBar);
                masterGrooveDataset.push_back(grooveBar);
            }
        }

        backgroundProgress.store((double)i / (double)totalFiles);
    }

    juce::Logger::writeToLog("Batch Complete. Bars processed: " + juce::String(masterRhythmDataset.size()));
    prepareTrainingTensors();
}

void ModelTrainerThread::prepareTrainingTensors()
{
    if (masterRhythmDataset.empty()) return;

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    int64_t numRows = (int64_t)masterRhythmDataset.size();

    std::vector<float> flatRhythm;
    for (const auto& row : masterRhythmDataset)
        flatRhythm.insert(flatRhythm.end(), row.begin(), row.end());
    trainingRhythmTensor = torch::from_blob(flatRhythm.data(), {numRows, 16}, options).clone();

    std::vector<float> flatGroove;
    for (const auto& row : masterGrooveDataset)
        flatGroove.insert(flatGroove.end(), row.begin(), row.end());
    trainingGrooveTensor = torch::from_blob(flatGroove.data(), {numRows, 16}, options).clone();
}

void ModelTrainerThread::estimateLatentDensity()
{
    if (trainingRhythmTensor.numel() == 0) return;

    model.eval();
    torch::NoGradGuard noGrad;

    auto latentSpace = model.encoder->forward(trainingRhythmTensor);

    latentMeans = torch::mean(latentSpace, 0);
    latentStdDevs = torch::std(latentSpace, 0);

    auto minStd = torch::full_like(latentStdDevs, 0.15f);
    latentStdDevs = torch::max(latentStdDevs, minStd);

    densityEstimated = true;

    juce::Logger::writeToLog("Ex-Post Density Estimation Complete.");
    std::cout << "Latent Dim 0 -> Mean: " << latentMeans[0].item<float>()
              << " Std: " << latentStdDevs[0].item<float>() << std::endl;
}

void ModelTrainerThread::saveDataset(const juce::File& outputFile)
{
    if (trainingRhythmTensor.numel() == 0 || trainingGrooveTensor.numel() == 0)
        return;

    try
    {
        torch::serialize::OutputArchive archive;
        archive.write("rhythm", trainingRhythmTensor);
        archive.write("groove", trainingGrooveTensor);
        archive.save_to(outputFile.getFullPathName().toStdString());

        juce::Logger::writeToLog("Dual-Model Dataset saved successfully: " + outputFile.getFileName());
    }
    catch (const std::exception& e)
    {
        juce::Logger::writeToLog("Save Error: " + juce::String(e.what()));
    }
}

void ModelTrainerThread::loadDataset(const juce::File& inputFile)
{
    try
    {
        torch::serialize::InputArchive archive;
        archive.load_from(inputFile.getFullPathName().toStdString());

        archive.read("rhythm", trainingRhythmTensor);
        archive.read("groove", trainingGrooveTensor);

        juce::Logger::writeToLog("Dataset loaded successfully.");
        estimateLatentDensity(); // Automatically re-estimate density upon load to restore visual state
    }
    catch (const std::exception& e)
    {
        juce::Logger::writeToLog("Load Error: " + juce::String(e.what()));
    }
}
