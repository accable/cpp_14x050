// Code is taken from exercise_4.cpp and modified accordingly
// Oh and we modified the naming conventions to make it proper lol. Will refactor this and other codes soon.
// Some of the coding habits were taken from https://github.com/pytorch/examples/blob/main/cpp/mnist/mnist.cpp
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>  // For pi


// Building the model
// This is the shallow one
// struct Net : torch::nn::Module {
//     Net(): 
//         fc1(2, 128),
//         fc2(128, 2) {
    
//     register_module("fc1", fc1);
//     register_module("fc2", fc2);
//     }

//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(fc1->forward(x));
//         x = fc2->forward(x);
//         return x;
//     }

//     torch::nn::Linear fc1;
//     torch::nn::Linear fc2;
// };


// This is the deep one
struct Net : torch::nn::Module {
    Net(): 
        fc1(2, 4),
        fc2(4, 8),
        fc3(8, 16),
        fc4(16, 32),
        fc5(32, 64),
        fc6(64, 128),
        fc7(128, 2) {
    
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
    register_module("fc4", fc4);
    register_module("fc5", fc5);
    register_module("fc6", fc6);
    register_module("fc7", fc7);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::relu(fc3->forward(x));
        x = torch::relu(fc4->forward(x));
        x = torch::relu(fc5->forward(x));
        x = torch::relu(fc6->forward(x));
        x = fc7->forward(x);
        return x;
    }

    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;
    torch::nn::Linear fc4;
    torch::nn::Linear fc5;
    torch::nn::Linear fc6;
    torch::nn::Linear fc7;
};


// Generating the disc set
// Since the set is tensor, we use std::pair and declare it as torch::Tensor
std::pair<torch::Tensor, torch::Tensor> generate_disc_set(const int size){
    torch::Tensor input = torch::empty({size, 2}).uniform_(-1, 1);
    torch::Tensor target = input.pow(2).sum(1).sub(2/ M_PI).sign().add(1).div(2).to(torch::kLong);

    return std::make_pair(input, target);
}


// Train model function (more cleaner, I suppose)
void train_model(Net& model, const torch::Tensor& train_input, const torch::Tensor& train_target){

    // Network parameters and such 
    const float eta = 1e-1f;
    const int mini_batch_size = 100;
    const int epochs = 250;

    // Model stuff + optimizer
    model.train();  // Set model to train mode
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(eta));  // SGD w/ lr 1e-1f

    // Training loop (for 100 epochs w/ mini-batches)
    for(int i = 0; i < epochs; ++i){  // Epochs
        for(int k = 0; k < train_input.size(0); k += mini_batch_size){  // Mini-batching loop

            auto output = model.forward(train_input.narrow(0, k, mini_batch_size));
            auto loss = torch::nn::functional::cross_entropy(output, train_target.narrow(0, k, mini_batch_size));
            
            // Loss backprop
            model.zero_grad();
            loss.backward();
            optimizer.step();
        }  
    }
}  


// Evaluating the model w.r.t test set
int compute_nb_errors(Net& model, const torch::Tensor input, const torch::Tensor target){

    // Some parameters
    int nb_errors = 0;
    const int mini_batch_size = 100;

    for(int k = 0; k < input.size(0); k += mini_batch_size){  // Mini-batching loop

        auto output = model.forward(input.narrow(0, k, mini_batch_size));
        auto predicted_class = std::get<1>(output.max(1));

        for(int b = 0; b < mini_batch_size; ++b){
            if(target[b + k].item<int>() != predicted_class[b].item<int>()) {  // Might due for a revisit
                nb_errors += 1;
            }
        }
    }  

    return nb_errors;
}


// The main function (reused from exercise_4.cpp)
auto main() -> int {

    torch::manual_seed(42);
    
    std::vector<torch::Tensor> train_images, train_labels;
    std::vector<torch::Tensor> test_images, test_labels;

    // Dataset
    auto [train_input, train_target] = generate_disc_set(1000);
    auto [test_input, test_target] = generate_disc_set(1000);

    // Normalize
    auto mu = train_input.mean();
    auto std = train_input.std();
    train_input.sub(mu).div(std);
    test_input.sub(mu).div(std);

    std::cout << "Done loading dataset!" << std::endl;

    // Since the loop uses standard deviations, we need to do this manually
    // Thanks cpp
    std::vector<float> stds = {-1, 1e-3, 1e-2, 1e-1, 1e-0, 1e1};

    // Model declarations
    Net model;

    for(float std : stds){
        if(std > 0){
            torch::NoGradGuard no_grad;
            for(auto& p : model.parameters()){
                p.normal_(0, std);
            }
        }

        std::cout << "Using std: " << std << std::endl;
        train_model(model, train_input, train_target);  
        int train_error = 100.0f * compute_nb_errors(model, train_input, train_target) / train_input.size(0);  // Train
        int test_error = 100.0f * compute_nb_errors(model, test_input, test_target) / test_input.size(0);  // Test
        
        std::cout << "Train Error: " << std::fixed << std::setprecision(2) 
        << train_error << "% " << "Test Error: " << std::fixed << std::setprecision(2) 
        << test_error << "% " << std::endl;
    }

    return 0;
}
