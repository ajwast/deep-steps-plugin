#pragma once

#include <torch/torch.h>

struct Autoencoder : torch::nn::Module {
    Autoencoder()
        : encoder(torch::nn::Sequential(
              torch::nn::Linear(16, 8),
              torch::nn::BatchNorm1d(8),
              torch::nn::ReLU(),
              torch::nn::Linear(8, 4),
              torch::nn::BatchNorm1d(4),
              torch::nn::ReLU())),
          decoder(torch::nn::Sequential(
              torch::nn::Linear(4, 8),
              torch::nn::BatchNorm1d(8),
              torch::nn::ReLU(),
              torch::nn::Linear(8, 16),
              torch::nn::Sigmoid()))
    {
        register_module("encoder", encoder);
        register_module("decoder", decoder);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = encoder->forward(x);
        x = decoder->forward(x);
        return x;
    }

    torch::Tensor decode(torch::Tensor x) {
        return decoder->forward(x);
    }

    /**
     * Trains the model on a provided dataset.
     * @param data A tensor of shape {BatchSize, 16} containing training patterns.
     * @param epochs Number of times to iterate over the dataset.
     * @param learningRate How fast the model learns (default 0.01).
     */
    void trainModel(torch::Tensor data, int epochs, double learningRate = 0.01) {
        // 1. Set model to training mode (important for BatchNorm!)
        this->train();

        // 2. Setup Optimizer (Adam is a standard choice)
        torch::optim::Adam optimizer(this->parameters(), torch::optim::AdamOptions(learningRate));

        // 3. Training Loop
        for (int epoch = 0; epoch < epochs; ++epoch) {
            optimizer.zero_grad();               // Reset gradients
            
            torch::Tensor prediction = forward(data); // Forward pass
            
            // 4. Calculate Loss
            // Binary Cross Entropy is ideal for Sigmoid outputs (0 to 1)
            torch::Tensor loss = torch::nn::functional::binary_cross_entropy(prediction, data);
            
            // 5. Backpropagation
            loss.backward();
            optimizer.step();

            // Optional: Log progress every 10 epochs
            if (epoch % 10 == 0) {
                std::cout << "Epoch: " << epoch << " | Loss: " << loss.item<float>() << std::endl;
            }
        }

        // 6. Set back to evaluation mode after training
        this->eval();
    }
    void trainBatch(const torch::Tensor& batch, torch::optim::Adam& optimizer) {
        this->train();
        optimizer.zero_grad();

        torch::Tensor prediction = forward(batch);
        
        // In an autoencoder, the input (batch) IS the target
        torch::Tensor loss = torch::nn::functional::binary_cross_entropy(prediction, batch);

        loss.backward();
        optimizer.step();
    }

        torch::nn::Sequential encoder;
        torch::nn::Sequential decoder;
    };
