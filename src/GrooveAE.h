#pragma once

#include <torch/torch.h>
//
//struct Autoencoder : torch::nn::Module {
//    Autoencoder()
//        : encoder(torch::nn::Sequential(
//              torch::nn::Linear(16, 8),
//              torch::nn::BatchNorm1d(8),
//              torch::nn::ReLU(),
//              torch::nn::Linear(8, 4),
//              torch::nn::BatchNorm1d(4),
//              torch::nn::ReLU())),
//          decoder(torch::nn::Sequential(
//              torch::nn::Linear(4, 8),
//              torch::nn::BatchNorm1d(8),
//              torch::nn::ReLU(),
//              torch::nn::Linear(8, 16),
//              torch::nn::Sigmoid()))
//    {
//        register_module("encoder", encoder);
//        register_module("decoder", decoder);
//    }
//
//    torch::Tensor forward(torch::Tensor x) {
//        x = encoder->forward(x);
//        x = decoder->forward(x);
//        return x;
//    }
//
//    torch::Tensor decode(torch::Tensor x) {
//        return decoder->forward(x);
//    }
//
//    /**
//     * Trains the model on a provided dataset.
//     * @param data A tensor of shape {BatchSize, 16} containing training patterns.
//     * @param epochs Number of times to iterate over the dataset.
//     * @param learningRate How fast the model learns (default 0.01).
//     */
//
//    void trainBatch(const torch::Tensor& batch, torch::optim::Adam& optimizer) {
//        this->train();
//        optimizer.zero_grad();
//
//        torch::Tensor prediction = forward(batch);
//        
//        // In an autoencoder, the input (batch) IS the target
////        torch::Tensor loss = torch::nn::functional::binary_cross_entropy(prediction, batch);
//        
//        torch::Tensor loss = torch::nn::functional::mse_loss(prediction, batch)
//
//        loss.backward();
//        optimizer.step();
//    }
//
//        torch::nn::Sequential encoder;
//        torch::nn::Sequential decoder;
//    };
//
