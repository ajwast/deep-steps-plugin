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
                std::function<void()> onChange)
        : apvts(vts), xID(paramID_X), yID(paramID_Y), onParameterChanged(onChange)
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
        // 1. Draw Background/Heatmap
        if (heatmap.isValid())
            g.drawImageWithin(heatmap, 0, 0, getWidth(), getHeight(), juce::RectanglePlacement::fillDestination);
        else
        {
            g.setColour(juce::Colours::black.withAlpha(0.3f));
            g.fillRoundedRectangle(getLocalBounds().toFloat(), 5.0f);
        }

        // 2. Draw Grid
        g.setColour(juce::Colours::white.withAlpha(0.1f));
        g.drawRect(getLocalBounds(), 1);
        g.drawLine(getWidth() / 2.0f, 0, getWidth() / 2.0f, (float)getHeight());
        g.drawLine(0, getHeight() / 2.0f, (float)getWidth(), getHeight() / 2.0f);

        // 3. Draw Puck
        auto x = apvts.getRawParameterValue(xID)->load();
        auto y = apvts.getRawParameterValue(yID)->load();

        auto puckPos = valueToPoint(x, y);
        
        // 3. Draw Crosshairs
        g.setColour(juce::Colours::white.withAlpha(0.2f));
        g.drawVerticalLine(puckPos.getX(), 0, (float)getHeight());
        g.drawHorizontalLine(puckPos.getY(), 0, (float)getWidth());

        // 4. Draw Puck
        g.setColour(juce::Colours::white);
        g.fillEllipse(puckPos.getX() - 5, puckPos.getY() - 5, 10, 10);
        g.drawEllipse(puckPos.getX() - 7, puckPos.getY() - 7, 14, 14, 1.0f);
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

    // Attachments need a slider to work with, even if we don't show it
    // Actually, for custom components, it's better to use ParameterAttachment or 
    // just update the parameter directly as I did in updateParametersFromMouse.
    // But SliderAttachment handles the undo/redo and host notification nicely.
    juce::Slider dummySliderX, dummySliderY;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> xAttachment, yAttachment;

    juce::Image heatmap;

    class Listener : public juce::AudioProcessorValueTreeState::Listener
    {
    public:
        Listener(LatentXYPad& p) : owner(p) {}
        void parameterChanged(const juce::String& parameterID, float newValue) override
        {
            owner.repaint();
        }
    private:
        LatentXYPad& owner;
    };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LatentXYPad)
};
