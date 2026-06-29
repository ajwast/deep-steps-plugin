#include "CustomLookAndFeel.h"
#include <JuceHeader.h>

CustomLookAndFeel::CustomLookAndFeel()
{

    
    // Sliders
    setColour(juce::Slider::trackColourId, juce::Colour(0xff252936));
    setColour(juce::Slider::backgroundColourId, juce::Colour(0xff12131a));
    setColour(juce::Slider::thumbColourId, juce::Colour(0xff00ffcc)); // retro neon cyan fader
    setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colour(0xff2d3142));
    setColour(juce::Slider::rotarySliderFillColourId, juce::Colour(0xff00ffcc));

    // Text / Labels
    setColour(juce::Label::textColourId, juce::Colour(0xffe2e8f0));

    // Combo Box / Popup Menu
    setColour(juce::ComboBox::backgroundColourId, juce::Colour(0xff1e2230));
    setColour(juce::ComboBox::textColourId, juce::Colour(0xffe0e6ed));
    setColour(juce::ComboBox::outlineColourId, juce::Colour(0xff4a5568));
    setColour(juce::ComboBox::arrowColourId, juce::Colour(0xffa0aec0));

    setColour(juce::PopupMenu::backgroundColourId, juce::Colour(0xff1e2230));
    setColour(juce::PopupMenu::textColourId, juce::Colour(0xffe0e6ed));
    setColour(juce::PopupMenu::highlightedBackgroundColourId, juce::Colour(0xff2b6cb0));
    setColour(juce::PopupMenu::highlightedTextColourId, juce::Colours::white);

    // Buttons
    setColour(juce::TextButton::buttonColourId, juce::Colour(0xff1c1e27));
    setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xff1c1e27));
    setColour(juce::TextButton::textColourOffId, juce::Colour(0xffe2e8f0));
    setColour(juce::TextButton::textColourOnId, juce::Colour(0xff00ffcc));
}

CustomLookAndFeel::~CustomLookAndFeel()
{
}

void CustomLookAndFeel::loadTypefacesIfNeeded()
{
    if (boldTypeface == nullptr && BinaryData::LeagueSpartanBold_ttfSize > 100) {
        boldTypeface = juce::Typeface::createSystemTypefaceFor(
            BinaryData::LeagueSpartanBold_ttf, BinaryData::LeagueSpartanBold_ttfSize);
    }

    if (regularTypeface == nullptr && BinaryData::LeagueSpartanRegular_ttfSize > 100) {
        regularTypeface = juce::Typeface::createSystemTypefaceFor(
            BinaryData::LeagueSpartanRegular_ttf, BinaryData::LeagueSpartanRegular_ttfSize);
    }

    if (monoTypeface == nullptr && BinaryData::LeagueMonoBold_otfSize > 100) {
        monoTypeface = juce::Typeface::createSystemTypefaceFor(
            BinaryData::LeagueMonoBold_otf,
            static_cast<size_t>(BinaryData::LeagueMonoBold_otfSize));
    }
}

juce::Typeface::Ptr CustomLookAndFeel::getTypefaceForFont(const juce::Font& f)
{
    loadTypefacesIfNeeded();

    if (f.isBold() && boldTypeface != nullptr)
        return boldTypeface;

    if (regularTypeface != nullptr)
        return regularTypeface;

    return juce::LookAndFeel_V4::getTypefaceForFont(f);
}

juce::Font CustomLookAndFeel::getCustomFont(float size, bool isBold)
{
    loadTypefacesIfNeeded();

    auto typeface = isBold ? boldTypeface : regularTypeface;

    // Build the Font object explicitly from the memory typeface, NOT the string name
    if (typeface != nullptr)
        return juce::Font(typeface).withHeight(size);

    // Fallback just in case
    return juce::Font(size).withStyle(isBold ? juce::Font::bold : juce::Font::plain);
}

juce::Font CustomLookAndFeel::getMonoFont(float size)
{
    loadTypefacesIfNeeded();

    if (monoTypeface != nullptr)
        return juce::Font(monoTypeface).withHeight(size);

    // Fallback to bold system font
    return juce::Font(size).withStyle(juce::Font::bold);
}



void CustomLookAndFeel::drawLinearSlider(juce::Graphics& g, int x, int y, int width, int height,
                                         float sliderPos, float minSliderPos, float maxSliderPos,
                                         const juce::Slider::SliderStyle style, juce::Slider& slider)
{
    // Vertical Sequencer Pitch Slider Drawing (looks like vintage physical console fader)
    if (slider.isVertical())
    {
        auto trackWidth = 4.0f;
        float trackX = x + width * 0.5f - trackWidth * 0.5f;

        // Draw fader track groove background
        g.setColour(slider.findColour(juce::Slider::backgroundColourId));
        g.fillRoundedRectangle(trackX, y, trackWidth, (float)height, trackWidth * 0.5f);

        // Draw physical tick markings beside the track
        g.setColour(juce::Colours::white.withAlpha(0.08f));
        int numTicks = 8;
        for (int i = 0; i <= numTicks; ++i)
        {
            float tickY = y + i * (height / (float)numTicks);
            g.drawHorizontalLine((int)tickY, x + width * 0.25f, x + width * 0.75f);
        }

        // Draw active level track fill
        g.setColour(slider.findColour(juce::Slider::trackColourId).withAlpha(0.3f));
        g.fillRoundedRectangle(trackX, sliderPos, trackWidth, height - (sliderPos - y), trackWidth * 0.5f);

        // Draw rectangular fader thumb (retro console fader cap)
        float thumbHeight = 12.0f;
        float thumbWidth = width * 0.75f;
        float thumbX = x + width * 0.5f - thumbWidth * 0.5f;
        float thumbY = sliderPos - thumbHeight * 0.5f;

        // Fader cap background (metallic look with dark borders)
        g.setColour(juce::Colour(0xff2d3142));
        g.fillRect(thumbX, thumbY, thumbWidth, thumbHeight);

        // Solid neon block detail on the fader cap sides
        g.setColour(slider.findColour(juce::Slider::thumbColourId));
        g.fillRect(thumbX, thumbY, 3.0f, thumbHeight);
        g.fillRect(thumbX + thumbWidth - 3.0f, thumbY, 3.0f, thumbHeight);

        // Fader cap outline border
        g.setColour(juce::Colours::black.withAlpha(0.5f));
        g.drawRect(thumbX, thumbY, thumbWidth, thumbHeight, 1.0f);

        // Center neon indicator line on fader cap
        g.setColour(slider.findColour(juce::Slider::thumbColourId));
        g.drawHorizontalLine((int)(thumbY + thumbHeight * 0.5f), thumbX + 3.0f, thumbX + thumbWidth - 3.0f);
    }
    else
    {
        // Horizontal Slider Drawing
        auto trackHeight = 4.0f;
        float trackY = y + height * 0.5f - trackHeight * 0.5f;

        g.setColour(slider.findColour(juce::Slider::backgroundColourId));
        g.fillRoundedRectangle((float)x, trackY, (float)width, trackHeight, trackHeight * 0.5f);

        g.setColour(slider.findColour(juce::Slider::trackColourId));
        g.fillRoundedRectangle((float)x, trackY, sliderPos - x, trackHeight, trackHeight * 0.5f);

        // Draw rectangular slider thumb
        float thumbWidth = 12.0f;
        float thumbHeight = height * 0.75f;
        float thumbX = sliderPos - thumbWidth * 0.5f;
        float thumbY = y + height * 0.5f - thumbHeight * 0.5f;

        g.setColour(juce::Colour(0xff2d3142));
        g.fillRect(thumbX, thumbY, thumbWidth, thumbHeight);

        g.setColour(slider.findColour(juce::Slider::thumbColourId));
        g.fillRect(thumbX, thumbY, thumbWidth, 3.0f);
        g.fillRect(thumbX, thumbY + thumbHeight - 3.0f, thumbWidth, 3.0f);

        g.setColour(juce::Colours::black.withAlpha(0.5f));
        g.drawRect(thumbX, thumbY, thumbWidth, thumbHeight, 1.0f);

        g.setColour(slider.findColour(juce::Slider::thumbColourId));
        g.drawVerticalLine((int)(thumbX + thumbWidth * 0.5f), thumbY + 3.0f, thumbY + thumbHeight - 3.0f);
    }
}

void CustomLookAndFeel::drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                                         float sliderPosProportionally, float rotaryStartAngle, float rotaryEndAngle,
                                         juce::Slider& slider)
{
    // Retro synthesizer knob dial drawing
    auto bounds = juce::Rectangle<int>(x, y, width, height).toFloat().reduced(6);
    auto radius = juce::jmin(bounds.getWidth(), bounds.getHeight()) / 2.0f;
    auto toX = bounds.getCentreX();
    auto toY = bounds.getCentreY();

    // 1. Dial Background
    g.setColour(juce::Colour(0xff12131a));
    g.fillEllipse(toX - radius, toY - radius, radius * 2.0f, radius * 2.0f);

    g.setColour(juce::Colour(0xff252936));
    g.drawEllipse(toX - radius, toY - radius, radius * 2.0f, radius * 2.0f, 1.5f);

    // 2. Active Value Arc (Neon dial ring)
    auto angle = rotaryStartAngle + sliderPosProportionally * (rotaryEndAngle - rotaryStartAngle);
    
    if (slider.isEnabled())
    {
        juce::Path arcPath;
        arcPath.addCentredArc(toX, toY, radius - 3.0f, radius - 3.0f, 0.0f, rotaryStartAngle, angle, true);
        
        g.setColour(slider.findColour(juce::Slider::rotarySliderFillColourId));
        g.strokePath(arcPath, juce::PathStrokeType(3.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    }

    // 3. Inner Dial Cap
    float innerRadius = radius - 7.0f;
    g.setColour(juce::Colour(0xff1c1e27));
    g.fillEllipse(toX - innerRadius, toY - innerRadius, innerRadius * 2.0f, innerRadius * 2.0f);

    g.setColour(juce::Colour(0xff2d3142));
    g.drawEllipse(toX - innerRadius, toY - innerRadius, innerRadius * 2.0f, innerRadius * 2.0f, 1.0f);

    // 4. Position tick/marker pointing to current value
    juce::Path pointer;
    float pointerLength = innerRadius - 1.0f;
    float pointerThickness = 2.5f;
    pointer.addRectangle(-pointerThickness * 0.5f, -innerRadius, pointerThickness, pointerLength);
    pointer.applyTransform(juce::AffineTransform::rotation(angle).translated(toX, toY));

    g.setColour(slider.isEnabled() ? juce::Colours::white : juce::Colours::grey);
    g.fillPath(pointer);
}

void CustomLookAndFeel::drawButtonBackground(juce::Graphics& g, juce::Button& button,
                                             const juce::Colour& backgroundColour,
                                             bool shouldDrawButtonAsHighlighted,
                                             bool shouldDrawButtonAsDown)
{
    auto bounds = button.getLocalBounds().toFloat();

    // Tactile retro button layout
    g.setColour(button.findColour(juce::TextButton::buttonColourId));
    g.fillRoundedRectangle(bounds, 3.0f);

    juce::Colour borderCol = shouldDrawButtonAsDown ? juce::Colour(0xff00ffcc) : 
                            (shouldDrawButtonAsHighlighted ? juce::Colour(0xffff9f0a) : juce::Colour(0xff4a5568));
    
    g.setColour(borderCol);
    g.drawRoundedRectangle(bounds, 3.0f, 1.5f);

    if (shouldDrawButtonAsDown)
    {
        g.setColour(borderCol.withAlpha(0.12f));
        g.fillRoundedRectangle(bounds.reduced(1.0f), 3.0f);
    }
}

void CustomLookAndFeel::drawButtonText(juce::Graphics& g, juce::TextButton& button,
                                       bool shouldDrawButtonAsHighlighted,
                                       bool shouldDrawButtonAsDown)
{
    g.setFont(getTextButtonFont(button, button.getHeight()));

    juce::Colour textCol = shouldDrawButtonAsDown ? juce::Colour(0xff00ffcc) : 
                           (shouldDrawButtonAsHighlighted ? juce::Colours::white : button.findColour(juce::TextButton::textColourOffId));
    g.setColour(textCol);

    g.drawFittedText(button.getButtonText(), button.getLocalBounds(),
                     juce::Justification::centred, 2);
}

void CustomLookAndFeel::drawProgressBar(juce::Graphics& g, juce::ProgressBar& progressBar,
                                        int width, int height, double progress, const juce::String& textToShow)
{
    // Retro segmented pixel green progress bar
    g.setColour(juce::Colour(0xff12131a));
    g.fillRoundedRectangle(0.0f, 0.0f, (float)width, (float)height, 3.0f);

    g.setColour(juce::Colour(0xff2d3142));
    g.drawRoundedRectangle(0.0f, 0.0f, (float)width, (float)height, 3.0f, 1.5f);

    if (progress > 0.0)
    {
        int progressWidth = static_cast<int>(width * progress);
        g.setColour(juce::Colour(0xff00ff66)); // vibrant retro neon green

        int segmentWidth = 6;
        int gap = 2;
        int currentX = 2;
        int maxProgressX = progressWidth - 2;

        while (currentX < maxProgressX && currentX < width - 2)
        {
            int blockW = juce::jmin(segmentWidth, maxProgressX - currentX);
            g.fillRect(currentX, 2, blockW, height - 4);
            currentX += segmentWidth + gap;
        }
    }

    if (textToShow.isNotEmpty())
    {
        g.setFont(12.0f);
        g.setColour(juce::Colours::white);
        g.drawFittedText(textToShow, 0, 0, width, height, juce::Justification::centred, 1);
    }
}

void CustomLookAndFeel::drawComboBox(juce::Graphics& g, int width, int height, bool isButtonDown,
                                     int buttonX, int buttonY, int buttonWidth, int buttonHeight,
                                     juce::ComboBox& box)
{
    auto cornerSize = 3.0f;
    auto bounds = juce::Rectangle<int>(0, 0, width, height).toFloat();

    g.setColour(box.findColour(juce::ComboBox::backgroundColourId));
    g.fillRoundedRectangle(bounds, cornerSize);

    g.setColour(box.findColour(juce::ComboBox::outlineColourId));
    g.drawRoundedRectangle(bounds, cornerSize, 1.2f);

    // Dropdown arrow
    auto arrowX = width - 18;
    auto arrowY = height / 2 - 2;
    g.setColour(box.findColour(juce::ComboBox::arrowColourId));

    juce::Path p;
    p.startNewSubPath((float)arrowX, (float)arrowY);
    p.lineTo((float)(arrowX + 8), (float)arrowY);
    p.lineTo((float)(arrowX + 4), (float)(arrowY + 5));
    p.closeSubPath();
    g.fillPath(p);
}

void CustomLookAndFeel::positionComboBoxText(juce::ComboBox& box, juce::Label& labelToPosition)
{
    labelToPosition.setBounds(5, 1, box.getWidth() - 25, box.getHeight() - 2);
}

