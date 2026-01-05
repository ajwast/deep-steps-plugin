#pragma once

#include <torch/torch.h>

struct GaussianGrooveModel : torch::nn::Module {
    GaussianGrooveModel() {
        // Shared layers to understand the rhythm context
        hidden = register_module("hidden", torch::nn::Sequential(
            torch::nn::Linear(16, 64),
            torch::nn::ReLU(),
            torch::nn::Linear(64, 64),
            torch::nn::ReLU()
        ));

        // Output heads
        mu_head = register_module("mu_head", torch::nn::Linear(64, 16));
        sigma_head = register_module("sigma_head", torch::nn::Linear(64, 16));
    }

    torch::nn::Sequential hidden;
    torch::nn::Linear mu_head{nullptr}, sigma_head{nullptr};

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        auto h = hidden->forward(x);
        
        // FIX: Using correct head names defined in constructor
        torch::Tensor mu_raw = mu_head->forward(h);
        torch::Tensor mu = torch::tanh(mu_raw); // Guaranteed -1.0 to 1.0

        torch::Tensor sigma_raw = sigma_head->forward(h);
        torch::Tensor sigma = torch::nn::functional::softplus(sigma_raw);

        return {mu, sigma};
    }
};

inline torch::Tensor calculateGaussianLoss(torch::Tensor mu, torch::Tensor sigma,
                                           torch::Tensor targets, torch::Tensor mask)
{
    // The Gaussian Log-Likelihood formula
    // We add a tiny epsilon (1e-6) to sigma to prevent log(0)
    auto variance = sigma.pow(2) + 1e-6;
    auto log_likelihood = -0.5 * torch::log(2 * M_PI * variance)
                          - 0.5 * (targets - mu).pow(2) / variance;

    // Apply the rhythm mask so we only care about steps where notes exist
    auto masked_ll = log_likelihood * mask;

    // Minimize Negative Log Likelihood
    return -masked_ll.sum() / (mask.sum() + 1e-8);
}