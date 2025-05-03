#include <torch/torch.h>
#include <iostream>
#include <vector>


// Where the dataroot would be (for dataset)
const char* kDataRoot =  "../dataset/MNIST/raw";  // This was run from build/


// Defining helper functions (loss functions, activations, and their derivative)
// Creating simple activation function call (tanh)
torch::Tensor sigma(const torch::Tensor& x){
    return torch::tanh(x);
}

// The derivative of the activation function
torch::Tensor dsigma(const torch::Tensor& x){
    return 4 * torch::pow(torch::exp(x) + torch::exp(-x), -2);
}

// Defining the loss function (MSE)
torch::Tensor loss(const torch::Tensor& v, const torch::Tensor& t){
    return torch::sum(torch::pow((v - t), 2));
}

// Defining the derivative of said loss function
torch::Tensor dloss(const torch::Tensor& v, const torch::Tensor t){
    return 2 * (v - t);
}


// Defining forward passes
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward_pass(
    const torch::Tensor& w1, const torch::Tensor& b1, 
    const torch::Tensor& w2, const torch::Tensor& b2,
    const torch::Tensor& x){

        auto x0 = x;
        auto s1 = torch::mv(w1, x0) + b1;
        auto x1 = sigma(s1);
        auto s2 = torch::mv(w2, x1) + b2;
        auto x2 = sigma(s2);

        return std::make_tuple(x0, s1, x1, s2, x2);

} 

// And the backward passes
void backward_pass(const torch::Tensor& w1, const torch::Tensor& b1,
    const torch::Tensor& w2, const torch::Tensor& b2,
    const torch::Tensor& t, const torch::Tensor& x,
    const torch::Tensor& x1, const torch::Tensor& s1,
    const torch::Tensor& x2, const torch::Tensor& s2,
    torch::Tensor& dl_dw1, torch::Tensor& dl_db1, 
    torch::Tensor& dl_dw2, torch::Tensor& dl_db2){

        auto x0 = x;
        auto dl_dx2 = dloss(x2, t);
        auto dl_ds2 = dsigma(s2) * dl_dx2;
        auto dl_dx1 = torch::mv(w2.t(), dl_ds2);
        auto dl_ds1 = dsigma(s1) * dl_dx1;

        dl_dw2.add_(dl_ds2.view({-1, 1}).mm(x1.view({1, -1})));
        dl_db2.add_(dl_ds2);
        dl_dw1.add_(dl_ds1.view({-1, 1}).mm(x0.view({1, -1})));
        dl_db1.add_(dl_ds1);

}



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

    // The whole thing + flattened + sliced by n_samples
    auto train_data = torch::cat(train_images).view({-1, 28 * 28}).slice(0, 0, n_samples);
    auto train_target = torch::cat(train_labels).slice(0, 0, n_samples);
    auto test_data = torch::cat(test_images).view({-1, 28 * 28}).slice(0, 0, n_samples);
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

    // The real deal...
    int nb_classes = train_target.size(1);
    int nb_train_samples = train_data.size(0);

    // Scaling targets to fit tanh better
    float zeta = 0.90f;
    train_target = train_target * zeta;
    test_target = test_target * zeta;

    // Network parameters (we skip the nb_hidden since we baked it directly to the main model function)
    float eta = 1e-1f / nb_train_samples;
    float epsilon = 1e-6f;

    // Initializing weights
    auto w1 = torch::empty({50, train_data.size(1)}).normal_(0, epsilon);
    auto w2 = torch::empty({nb_classes, 50}).normal_(0, epsilon);
    auto b1 = torch::empty({50}).normal_(0, epsilon);
    auto b2 = torch::empty({nb_classes}).normal_(0, epsilon);

    // Initializing backprop gradients
    auto dl_dw1 = torch::empty_like(w1);
    auto dl_dw2 = torch::empty_like(w2);
    auto dl_db1 = torch::empty_like(b1);
    auto dl_db2 = torch::empty_like(b2);

    // Training loop
    for(int i = 0; i < 1000; ++i){
        
        // Forward pass and back-propagate
        float acc_loss = 0;
        int nb_train_errors = 0;

        // Setting grad to zero
        dl_dw1.zero_();
        dl_dw2.zero_();
        dl_db1.zero_();
        dl_db2.zero_();

        for(int k = 0; k < nb_train_samples; ++k){
            auto [x0, s1, x1, s2, x2] = forward_pass(w1, b1, w2, b2, train_data[k]);
            auto pred = x2.argmax(0).item<int>();

            if(train_target.index({k, pred}).item<float>() < 0.5f){
                nb_train_errors++;
            }
            acc_loss += loss(x2, train_target[k]).item<float>();

            backward_pass(w1, b1, w2, b2, train_target[k],
                            x0, x1, s1, x2, s2, 
                            dl_dw1, dl_db1, dl_dw2, dl_db2);
        }

        // Gradient step
        w1 -= eta * dl_dw1;
        w2 -= eta * dl_dw2;
        b1 -= eta * dl_db1;
        b2 -= eta * dl_db2;

        // Test errors
        int nb_test_errors = 0;

        for(int n = 0; n < test_data.size(0); ++n){
            auto [_, __, ___, ____, x2] = forward_pass(w1, b1, w2, b2, test_data[n]);
            auto pred = x2.argmax(0).item<int>();

            if(test_target.index({n, pred}).item<float>() < 0.5f){
                nb_test_errors++;
            }
        }

        // Printing statistics
        float train_error_percent = (100.0f * nb_train_errors) / nb_train_samples;
        float test_error_percent = (100.0f * nb_test_errors) / test_data.size(0);

        std::cout << i << " acc_train_loss = " << acc_loss 
        << " acc_train_error = " << train_error_percent << "%" 
        << " test_error = " << test_error_percent << "%" << std::endl;
        
    }

    return 0;

}