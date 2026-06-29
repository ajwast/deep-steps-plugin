#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

class CustomLookAndFeel : public juce::LookAndFeel_V4
{
public:
    CustomLookAndFeel();
    ~CustomLookAndFeel() override;

    // Font override
    juce::Typeface::Ptr getTypefaceForFont(const juce::Font& f) override;
    juce::Font getCustomFont(float size, bool isBold = false);
    juce::Font getMonoFont(float size);

    // Linear Slider (Fader) drawing override
    void drawLinearSlider(juce::Graphics& g, int x, int y, int width, int height,
                          float sliderPos, float minSliderPos, float maxSliderPos,
                          const juce::Slider::SliderStyle style, juce::Slider& slider) override;

    // Rotary Slider (Knob) drawing override
    void drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                          float sliderPosProportionally, float rotaryStartAngle, float rotaryEndAngle,
                          juce::Slider& slider) override;

    // TextButton drawing overrides
    void drawButtonBackground(juce::Graphics& g, juce::Button& button,
                              const juce::Colour& backgroundColour,
                              bool shouldDrawButtonAsHighlighted,
                              bool shouldDrawButtonAsDown) override;

    void drawButtonText(juce::Graphics& g, juce::TextButton& button,
                        bool shouldDrawButtonAsHighlighted,
                        bool shouldDrawButtonAsDown) override;

    // ProgressBar drawing override
    void drawProgressBar(juce::Graphics& g, juce::ProgressBar& progressBar,
                         int width, int height, double progress, const juce::String& textToShow) override;

    // ComboBox drawing overrides
    void drawComboBox(juce::Graphics& g, int width, int height, bool isButtonDown,
                      int buttonX, int buttonY, int buttonWidth, int buttonHeight,
                      juce::ComboBox& box) override;

    void positionComboBoxText(juce::ComboBox& box, juce::Label& labelToPosition) override;

private:
    juce::Typeface::Ptr regularTypeface = nullptr;
    juce::Typeface::Ptr boldTypeface = nullptr;
    juce::Typeface::Ptr monoTypeface = nullptr;
    void loadTypefacesIfNeeded();
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CustomLookAndFeel)
};
