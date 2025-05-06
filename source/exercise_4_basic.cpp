// This is the 'embryo' version of the assignment, which is later modified on other versions of this file.
// Code is taken from exercise_3.cpp and modified accordingly
#include <torch/torch.h>
#include <iostream>
#include <vector>


// Where the dataroot would be (for dataset)
const char* kDataRoot =  "../dataset/MNIST/raw";  // This was run from build/


// Building the model (following https://github.com/pytorch/examples/blob/main/cpp/mnist/mnist.cpp)
struct Net : torch::nn::Module {
    Net(): 
        conv1(torch::nn::Conv2dOptions(1, 32, 5)),
        conv2(torch::nn::Conv2dOptions(32, 64, 5)),
        fc1(256, 200),
        fc2(200, 10) {
    
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 3, {3, 3}));  // Kernel and stride is 3
        x = torch::relu(torch::max_pool2d(conv2->forward(x), 2, {2, 2}));  // Kernel and stride is 2
        x = x.view({-1, 256});
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};


// Dataset one-hot-labelling
torch::Tensor one_hot_labels(const torch::Tensor& input, const torch::Tensor& target){
    auto temp = torch::zeros({target.size(0), target.max().item<int>() + 1},
                            torch::TensorOptions().dtype(input.dtype()));
    temp.scatter_(1, target.view({-1, 1}), 1.0);
    
    return temp;
}


// The main function (reused from exercise_2.cpp)
auto main() -> int {

    // We now add the normalization and one-hot encoded labels
    // We also limit it to 1000 for both test and train set
    torch::manual_seed(42);
    const int n_samples = 1000; // We don't want to sample a lot
    
    std::vector<torch::Tensor> train_images, train_labels;
    std::vector<torch::Tensor> test_images, test_labels;

    // Train set
    auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
                            .map(torch::data::transforms::Stack<>());


    // Test set
    auto test_dataset = torch::data::datasets::MNIST(kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                            .map(torch::data::transforms::Stack<>());

    // Dataloader and some constants
    auto train_loader = torch::data::make_data_loader(train_dataset, train_dataset.size().value());
    auto test_loader = torch::data::make_data_loader(test_dataset, test_dataset.size().value());

    const size_t train_dataset_size = train_dataset.size().value();
    const size_t test_dataset_size = test_dataset.size().value();

    // Since .data and .target does not exist on cpp, we need to loop every single data and stack it on our 
    // vector tensor object
    for (auto& batch : *train_loader) {
        train_images.push_back(batch.data);
        train_labels.push_back(batch.target);
    }

    for (auto& batch : *test_loader) {
        test_images.push_back(batch.data);
        test_labels.push_back(batch.target);
    }

    // The whole thing + sliced by n_samples *we removed the flattening)
    auto train_data = torch::cat(train_images).slice(0, 0, n_samples);
    auto train_target = torch::cat(train_labels).slice(0, 0, n_samples);
    auto test_data = torch::cat(test_images).slice(0, 0, n_samples);
    auto test_target = torch::cat(test_labels).slice(0, 0, n_samples);

    // One-hot labels 
    train_target = one_hot_labels(train_data, train_target);
    test_target = one_hot_labels(test_data, test_target);

    // Normalize
    auto mu = train_data.mean();
    auto std = train_data.std();
    train_data = train_data.sub(mu).div(std);
    test_data = test_data.sub(mu).div(std);

    std::cout << "Done loading dataset!" << std::endl;

    // Model declarations + set to train mode
    Net model;
    model.train();

    // Network parameters 
    const float eta = 1e-1f;
    const float epsilon = 1e-6f;
    const int mini_batch_size = 100;

    // Training loop (for 100 epochs w/ mini-batches)
    for(int i = 0; i < 100; ++i){  // Epochs

        int acc_loss = 0;

        for(int k = 0; k < train_data.size(0); k += mini_batch_size){  // Mini-batching loop

            auto output = model.forward(train_data.narrow(0, k, mini_batch_size));
            auto loss = torch::nn::functional::mse_loss(output, train_target.narrow(0, k, mini_batch_size));
            acc_loss += loss.item<float>();
            
            // Loss backprop
            model.zero_grad();
            loss.backward();

            // Updating parameters
            {
                torch::NoGradGuard no_grad;
                for(auto& param: model.parameters()){
                    param -= eta * param.grad();
                }
            }
        }  

        std::cout << "Epoch: " << i << ", acc_loss: " << acc_loss << std::endl;
    }

    return 0;

}