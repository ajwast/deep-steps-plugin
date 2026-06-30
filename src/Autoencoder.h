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
              torch::nn::Tanh())),
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



        torch::nn::Sequential encoder;
        torch::nn::Sequential decoder;
    };