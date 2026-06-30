#include "AudioAnalyzer.h"
#include <regex>
#include <cmath>

AudioAnalyzer::AudioAnalyzer()
{
    formatManager.registerBasicFormats();
}

AnalysisResult AudioAnalyzer::analyzeFile(const juce::File& file)
{
    AnalysisResult result;
    if (auto* reader = formatManager.createReaderFor(file))
    {
        double sampleRate = reader->sampleRate;
        juce::AudioSampleBuffer buffer((int)reader->numChannels, (int)reader->lengthInSamples);
        reader->read(&buffer, 0, (int)reader->lengthInSamples, 0, true, true);
        delete reader;

        std::vector<double> onsets;
        detectOnsets(buffer, sampleRate, onsets);
        
        result.bpm = findTempoFromAudio(file);
        
        segmentAudioFile(buffer, sampleRate, result.bpm, onsets, result.steps, result.shifts);
    }
    return result;
}

int AudioAnalyzer::findTempoFromAudio(const juce::File& file)
{
    int detectedBpm = 0;

    const juce::String fileName = file.getFileNameWithoutExtension();
    std::string nameStd = fileName.toStdString();

    // Try "[number] bpm" with optional underscores
    std::regex bpmRegex ("(\\d{2,3})[_\\s]*[Bb][Pp][Mm]");
    std::smatch match;

    if (std::regex_search(nameStd, match, bpmRegex))
    {
        int value = std::stoi(match[1].str());
        if (value >= 80 && value <= 170)
        {
            detectedBpm = value;
            return detectedBpm;
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
            return detectedBpm;
        }
    }

    return detectedBpm;
}

void AudioAnalyzer::detectOnsets(const juce::AudioSampleBuffer& loadedBuffer, double sampleRate, std::vector<double>& onsets)
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

void AudioAnalyzer::segmentAudioFile(const juce::AudioSampleBuffer& loadedBuffer, double sampleRate, int bpm,
                                     const std::vector<double>& onsets, std::vector<int>& steps, std::vector<float>& shifts)
{
    const int numSamples = loadedBuffer.getNumSamples();
    if (numSamples == 0 || bpm <= 0 || sampleRate <= 0.0)
        return;

    // === Grid sizes ===
    const double quarterNoteSamples = (60.0 / bpm) * sampleRate;
    int sixteenthNoteSamples = (int) std::round(quarterNoteSamples / 4.0);

    int numSixteenths = (int) std::ceil((double)numSamples / sixteenthNoteSamples);

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

            const double offset = onsetSample - centerSample;
            shifts[nearestIdx] = juce::jlimit(
                -1.0f, 1.0f,
                (float) (offset / half32ndSamples)
            );
        }
    }
}
