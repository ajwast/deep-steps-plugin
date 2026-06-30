#pragma once

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_dsp/juce_dsp.h>
#include <vector>

struct AnalysisResult
{
    int bpm = 0;
    std::vector<int> steps;     // Flattened binary trigger array (16th notes)
    std::vector<float> shifts;  // Flattened groove shifts
};

class AudioAnalyzer
{
public:
    AudioAnalyzer();
    ~AudioAnalyzer() = default;

    AnalysisResult analyzeFile(const juce::File& file);

private:
    void detectOnsets(const juce::AudioSampleBuffer& buffer, double sampleRate, std::vector<double>& outOnsets);
    int findTempoFromAudio(const juce::File& file);
    void segmentAudioFile(const juce::AudioSampleBuffer& buffer, double sampleRate, int bpm,
                          const std::vector<double>& onsets, std::vector<int>& outSteps, std::vector<float>& outShifts);

    juce::AudioFormatManager formatManager;
};
