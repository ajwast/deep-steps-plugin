#!/bin/sh
set -e
if test "$CONFIGURATION" = "Custom"; then :
  cd "/Users/alexwastnidge/Documents/Xcode Projs/Deep-steps-JUCE-plugin/build/JUCE/tools"
  make -f /Users/alexwastnidge/Documents/Xcode\ Projs/Deep-steps-JUCE-plugin/build/JUCE/tools/CMakeScripts/ReRunCMake.make
fi

