#include <torch/torch.h>  // "I don't want to use PyTorch" ahh mf
#include <iostream>
#include <vector>

// Where the dataroot would be (for dataset)
const char* kDataRoot =  "../dataset/MNIST/raw";  // This was run from build/


// Finding the nearest neighbor. Using the implementation, the used 
// distance is squared euclidian distance.
int64_t nearest_classification(const torch::Tensor& train_input, 
                                const torch::Tensor& train_target, 
                                const torch::Tensor& x){
                                
    // Calculating the squared distance
    torch::Tensor dist = (train_input - x).pow(2).sum(1).view(-1);

    // Finding the index of the minimum distance
    torch::Tensor min_result = std::get<1>(dist.min(0));

    // Returning the corresponding label
    return train_target[min_result.item<int64_t>()].item<int64_t>();
}


// Now we find the PCA basis and the mean
std::pair<torch::Tensor, torch::Tensor> PCA(const torch::Tensor& x){

    // Computing the mean
    torch::Tensor mean = x.mean(0);

    // Centering the data
    torch::Tensor b = x - mean;

    // Computing the eigenvectors
    // Using SVD since torch::linalg::eig does not exist
    auto svd = b.svd();
    torch::Tensor eigenvectors = std::get<2>(svd);

    return std::make_pair(mean, eigenvectors);
}


// Computing the number of errors
int compute_nb_errors(torch::Tensor& train_input,
                        torch::Tensor& train_target,
                        torch::Tensor& test_input,
                        torch::Tensor& test_target,
                        std::optional<torch::Tensor> mean = std::nullopt,
                        std::optional<torch::Tensor> proj = std::nullopt){
    
    // Creating a copy of the tensors
    auto train_copy = train_input.clone();
    auto test_copy = test_input.clone();

    if(mean){
        train_copy = train_copy - mean.value();
        test_copy = test_copy - mean.value();
    }

    if(proj){
        train_copy = train_copy.mm(proj.value().t());
        test_copy = test_copy.mm(proj.value().t());
    }

    int nb_errors = 0;

    for(int64_t i = 0; i < test_copy.size(0); i++){
        auto predicted = nearest_classification(train_copy, train_target, test_copy[i]);

        if(predicted != test_target[i].item<int64_t>()){
            nb_errors += 1;
        }
    }

    return nb_errors;
}

auto main() -> int {

    // We omit the normalization and one-hot labels since it is not used
    // The data is flattened so we do it lol
    // We also limit it to 1000 for both test and train set
    torch::manual_seed(42);
    const int n_samples = 1000; // We don't want to sample a lot
    
    // Since the assignment does not consider dataloader, we need to get the data manually
    // Thanks libtorch
    std::vector<torch::Tensor> train_images, train_labels;
    std::vector<torch::Tensor> test_images, test_labels;

    // We are using MNIST for now
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

    std::cout << "Done loading dataset!" << std::endl;

    // Computing baseline nb_errors 
    auto nb_errors = compute_nb_errors(train_data, train_target, test_data, test_target);
    std::cout << "Baseline nb_errors: " << nb_errors << ", error: " << 100 * nb_errors / test_data.size(0) << std::endl; 

    auto random_proj = torch::randn({100, train_data.size(1)});
    nb_errors = compute_nb_errors(train_data, train_target, test_data, test_target, std::nullopt, random_proj);
    std::cout << "Random proj nb_errors: " << nb_errors << ", error: " << 100 * nb_errors / test_data.size(0) << std::endl;

    auto [mean, basis] = PCA(train_data);

    for (int d : {100, 50, 10, 3}){
        
        std::cout << basis.slice(0,0,d).sizes() << std::endl;
        std::cout << train_data.sizes() << std::endl;

        nb_errors = compute_nb_errors(train_data, train_target, test_data, test_target, mean, basis.slice(0, 0, d));
        std::cout << "With PCA " << d << " nb_errors: " << nb_errors << ", error: " << 100 * nb_errors / test_data.size(0) << std::endl;
    }

    return 0;

}