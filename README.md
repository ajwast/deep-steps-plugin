# Deep Steps - DAW Plugin

<img alt="DS_GUI.png" src="assets/images/DS_GUI.png" width="500"/>

## Introduction

Deep Steps is a generative MIDI sequencer driven by user-trainable neural networks.
This re-implementation is a DAW plugin using the JUCE C++ framework and Torch.

The original Deep Steps project can be found [here](https://github.com/ajwast/DeepSteps). It was created by Alex Wastnidge as part of their Master's thesis for the Music, Communication and Technology programme at the University of Oslo. It's development was presented at the [*International Conference on AI and Musical Creativity 2024*](https://aimc2024.pubpub.org/pub/odrhfynm/release/1)

This plugin implementation is very much ***in development***. See the status of the project and the "To dos" below.

## What's New?

The DAW plugin (re)implementation makes several key improvements on the original project:

### DAW MIDI Step Sequencer Plugin
- Implemented as DAW plugin
- Pure C++ implementation with JUCE framework and Torch (libtorch)
- Sample-accurate MIDI timing and synchronisation with DAW.
- Custom JUCE GUI
- All functionality implemented in-plugin, including model training.
- Many instances of plugin can run at once.

### Step Generation Neural Network
- RAE-L2 neural network architecture ([Ghosh et al. (2019)](https://arxiv.org/abs/1903.12436)) is used for step generation.
- RAE-L2 Ex-Post Density Estimation allows actual data distribution within latent space to be used and visualised in X/Y pad heatmaps.
- Batch normalisation used as in [Ghose, A., Rashwan, A. &amp; Poupart, P. (2020)](https://proceedings.mlr.press/v124/ghose20a.html)

### Groove Generation 
- Micro-timings for groove is driven by a multi-head regression model which takes the current step generation as input.
- Groove model outputs mu and sigma for each step. Each step therefore has a Gaussian probability distribution for micro-timing offset.
- Groove as probability distribution inspired by [Wright (2008)](https://ccrma.stanford.edu/~matt/diss/Matthew-Wright-Dissertation.pdf) and the 'Beat Bin Model' by [Danielsen (2010,2023)](https://academic.oup.com/book/56186/chapter/443057660)
- Groove offsets are re-sampled from Gaussian distributions every bar to mimic human playing, i.e. the groove is slightly different every bar.

### User Training Data
- Re-implemented onset detection in-plugin with spectral flux and peak picking.
- "Groove" is encoded as a continuous timing offset of a 16th note between its neighboring 32nd notes.
- Saving and loading functionality of analysed and encoded training data.

## To Do
- Release build of VST3
- Add model weights as part of the Value Tree State (APVTS) for model recall.
- Presets implementation: saving and recall of entire plugin state
- Documentation (any)
- Auto-validation testing using [pluginval](https://github.com/Tracktion/pluginval)
- AU, AAX, LV2, CLAP support
- User-trainable recurrent neural network for pitch generation

## Installation

Check the Releases page for current pre-built plugins

### Building from source
To build the plugin yourself you will need:

- JUCE C++ framework
- Libtorch C++
- CMake
- An IDE (CLion, VSCode, Xcode etc.)

Create your own *CMakeLists.txt* file using the example file. Change the paths to JUCE and libtorch to your own directories and build the project.

