wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz -P ./MNIST/raw/
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz -P ./MNIST/raw/
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz -P ./MNIST/raw/
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz -P ./MNIST/raw/

gunzip ./MNIST/raw/*.gz