#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>

class LatentXYPad : public juce::Component,
                    public juce::AudioProcessorValueTreeState::Listener
{
public:
    LatentXYPad(juce::AudioProcessorValueTreeState& vts, 
                const juce::String& paramID_X, 
                const juce::String& paramID_Y,
                std::function<void()> onChange,
                juce::Colour puckColor = juce::Colours::white)
        : apvts(vts), xID(paramID_X), yID(paramID_Y), onParameterChanged(onChange), puckColour(puckColor)
    {
        xAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(apvts, xID, dummySliderX);
        yAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(apvts, yID, dummySliderY);

        // Listen for parameter changes to repaint the puck
        apvts.addParameterListener(xID, this);
        apvts.addParameterListener(yID, this);
    }

    ~LatentXYPad() override
    {
        apvts.removeParameterListener(xID, this);
        apvts.removeParameterListener(yID, this);
    }



    void paint(juce::Graphics& g) override
{
    auto bounds = getLocalBounds().toFloat();
    auto cornerSize = 3.0f; // Matches your ComboBox styling

        // 1. Draw Background/Heatmap
        if (heatmap.isValid())
        {
            juce::Graphics::ScopedSaveState state (g); // Save state before clipping

            juce::Path clipPath;
            clipPath.addRoundedRectangle(bounds, cornerSize);
            g.reduceClipRegion(clipPath); // Apply rounded corners mask

            g.drawImageWithin(heatmap, 0, 0, getWidth(), getHeight(), juce::RectanglePlacement::fillDestination);
            // The ScopedSaveState destructor will automatically remove the clip when it goes out of scope
        }
        else
        {
            g.setColour(juce::Colour(0xff12131a));
            g.fillRoundedRectangle(bounds, cornerSize);
        }


    // 2. Draw Vector Grid (Cyberpunk style)
    g.setColour(juce::Colour(0xff2d3142).withAlpha(0.5f)); // Slate blue, semi-transparent

    int numDivisions = 8;
    float stepX = bounds.getWidth() / numDivisions;
    float stepY = bounds.getHeight() / numDivisions;

    for (int i = 1; i < numDivisions; ++i)
    {
        // Draw vertical grid lines
        g.drawVerticalLine(static_cast<int>(stepX * i), 0.0f, bounds.getHeight());
        // Draw horizontal grid lines
        g.drawHorizontalLine(static_cast<int>(stepY * i), 0.0f, bounds.getWidth());
    }

    // Draw the main axis crosshairs slightly brighter
    g.setColour(juce::Colour(0xff4a5568).withAlpha(0.6f));
    g.drawVerticalLine(getWidth() / 2, 0.0f, bounds.getHeight());
    g.drawHorizontalLine(getHeight() / 2, 0.0f, bounds.getWidth());

    // 3. Draw The Outer Frame (Matches ComboBox)
    g.setColour(juce::Colour(0xff4a5568)); // Outline Color
    g.drawRoundedRectangle(bounds, cornerSize, 1.2f);

    // 4. Draw Puck
    auto x = apvts.getRawParameterValue(xID)->load();
    auto y = apvts.getRawParameterValue(yID)->load();
    auto puckPos = valueToPoint(x, y);

    // Draw Targeting Crosshairs tracking the puck
    g.setColour(puckColour.withAlpha(0.2f));
    g.drawVerticalLine(static_cast<int>(puckPos.getX()), 0.0f, bounds.getHeight());
    g.drawHorizontalLine(static_cast<int>(puckPos.getY()), 0.0f, bounds.getWidth());

    // Draw HUD Puck (Reticle)
    g.setColour(puckColour);
    g.fillEllipse(puckPos.getX() - 3.0f, puckPos.getY() - 3.0f, 6.0f, 6.0f); // Inner glowing core

    g.setColour(puckColour.withAlpha(0.7f));
    g.drawEllipse(puckPos.getX() - 8.0f, puckPos.getY() - 8.0f, 16.0f, 16.0f, 1.5f); // Outer targeting ring
}

    void mouseDown(const juce::MouseEvent& e) override
    {
        updateParametersFromMouse(e.getPosition());
    }

    void mouseDrag(const juce::MouseEvent& e) override
    {
        updateParametersFromMouse(e.getPosition());
    }

    void setHeatmap(const juce::Image& newHeatmap)
    {
        heatmap = newHeatmap;
        repaint();
    }

    // Needed for APVTS listener
    void parameterChanged(const juce::String& parameterID, float newValue) override
    {
        juce::MessageManager::callAsync([this] { repaint(); });
    }

private:
    void updateParametersFromMouse(juce::Point<int> pos)
    {
        auto bounds = getLocalBounds().toFloat();
        
        float normX = juce::jlimit(0.0f, 1.0f, (float)pos.getX() / bounds.getWidth());
        float normY = juce::jlimit(0.0f, 1.0f, 1.0f - ((float)pos.getY() / bounds.getHeight()));

        auto* xParam = apvts.getParameter(xID);
        auto* yParam = apvts.getParameter(yID);

        if (xParam) xParam->setValueNotifyingHost(normX);
        if (yParam) yParam->setValueNotifyingHost(normY);

        if (onParameterChanged)
            onParameterChanged();
        
        repaint();
    }

    juce::Point<float> valueToPoint(float xVal, float yVal)
    {
        auto rangeX = apvts.getParameterRange(xID);
        auto rangeY = apvts.getParameterRange(yID);

        float normX = rangeX.convertTo0to1(xVal);
        float normY = rangeY.convertTo0to1(yVal);

        return { normX * getWidth(), (1.0f - normY) * getHeight() };
    }

    juce::AudioProcessorValueTreeState& apvts;
    juce::String xID, yID;
    std::function<void()> onParameterChanged;
    juce::Colour puckColour;

    juce::Slider dummySliderX, dummySliderY;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> xAttachment, yAttachment;
    juce::Image heatmap;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LatentXYPad)
};
