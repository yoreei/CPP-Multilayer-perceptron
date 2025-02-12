#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <algorithm>
#include "Eigen/Dense"

#include <chrono>
using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

Time getTime() {
	return std::chrono::high_resolution_clock::now();
}
using Milli = std::chrono::duration<double, std::milli>;
using Seconds = std::chrono::duration<double, std::ratio<1>>;

using namespace Eigen;
using MlpMatrix = MatrixXf;
using MlpVectorf = VectorXf;
using MlpVectori = VectorXi;
using MlpRowVectorf = RowVectorXf;

// Helper function to swap endianess (if needed)
uint32_t swapEndian(uint32_t val) {
	return ((val >> 24) & 0xff) |
		((val << 8) & 0xff0000) |
		((val >> 8) & 0xff00) |
		((val << 24) & 0xff000000);
}
MlpMatrix relu(const MlpMatrix& x) {
	return x.array().max(0.0).matrix();
}

MlpMatrix softmax(const MlpMatrix& x) {
	// Compute the maximum for each row (result is an n x 1 column vector)
	MlpMatrix rowMax = x.rowwise().maxCoeff();
	// Subtract the row max from every element in that row for numerical stability.
	// We replicate the rowMax to the same number of columns as x.
	MlpMatrix x_stable = x - rowMax.replicate(1, x.cols());

	// Compute exponentials
	MlpMatrix exp_x = x_stable.array().exp();

	// Compute row-wise sums of the exponentials (n x 1 vector)
	MlpVectorf rowSum = exp_x.rowwise().sum();

	// Divide each element by the sum of its row.
	// The division is done column-wise using broadcasting via Eigen’s array interface.
	MlpMatrix sm = exp_x.array().colwise() / rowSum.array();

	return sm;
}

struct Dataset {
	// Default constructor.
	Dataset() = default;

	// Constructor that reads MNIST data and label files.
	Dataset(const std::string& dataFile, const std::string& labelFile) {
		std::cout << "loading " << dataFile << std::endl;
		std::ifstream dataIfstream(dataFile, std::ios::binary);
		if (!dataIfstream) {
			std::cerr << "Unable to open file: " << dataFile << std::endl;
			exit(-1);
		}

		uint32_t magicNumber = 0;
		uint32_t numImages = 0;
		// Read header: magic number, number of images, rows, and columns.
		dataIfstream.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
		dataIfstream.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
		dataIfstream.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
		dataIfstream.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

		// Convert from big-endian if needed.
		magicNumber = swapEndian(magicNumber);
		numImages = swapEndian(numImages);
		numRows = swapEndian(numRows);
		numCols = swapEndian(numCols);

		std::cout << "Magic Number: " << magicNumber << "\n";
		std::cout << "Number of Images: " << numImages << "\n";
		std::cout << "Rows: " << numRows << "\n";
		std::cout << "Columns: " << numCols << "\n";

		// Allocate data matrix: each row will be one flattened image.
		data.resize(numImages, numRows * numCols);

		// Read image data (numImages * numRows*numCols bytes).
		// We fill the matrix element‐by‐element.
		for (size_t imgId = 0; imgId < numImages; ++imgId) {
			for (size_t i = 0; i < numRows * numCols; ++i) {
				char byte;
				dataIfstream.read(&byte, sizeof(char));
				if (!dataIfstream) {
					std::cerr << "Error reading byte " << i << std::endl;
					exit(-1);
				}
				// Normalize the pixel value to [0,1] and store as float.
				data(imgId, i) = static_cast<unsigned char>(byte) / 255.0f;
			}
		}

		// Now load the labels.
		std::cout << "loading " << labelFile << std::endl;
		std::ifstream labelIfstream(labelFile, std::ios::binary);
		if (!labelIfstream) {
			std::cerr << "Unable to open file: " << labelFile << std::endl;
			exit(-1);
		}

		uint32_t numLabels = 0;
		labelIfstream.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
		labelIfstream.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
		magicNumber = swapEndian(magicNumber);
		numLabels = swapEndian(numLabels);

		std::cout << "Magic Number: " << magicNumber << "\n";
		std::cout << "Number of Labels: " << numLabels << "\n";

		if (numLabels != numImages) {
			std::cerr << "numLabels != numImages" << std::endl;
			exit(-1);
		}

		labels.resize(numImages);
		for (size_t i = 0; i < numImages; ++i) {
			char byte;
			labelIfstream.read(&byte, sizeof(char));
			if (!dataIfstream) {
				throw std::runtime_error("error reading bytes");
			}
			labels(i) = static_cast<int>(byte);
		}
	}

	// Convert a pixel (float) to a character for visualization.
	char charFromFloat(float f) const {
		if (f > 0.7f) {
			return '#';
		}
		else if (f > 0.4f) {
			return '!';
		}
		else if (f > 0.1f) {
			return '.';
		}
		else if (f >= 0) {
			return ' ';
		}
		else {
			throw std::runtime_error("wrong f");
		}
	}

	// Print out statistics and show the 40th image.
	void statistics() const {
		std::cout << "Data dim [" << data.rows() << ", " << data.cols() << "]" << std::endl;
		float minVal = data.minCoeff();
		float maxVal = data.maxCoeff();
		float sumVal = data.sum();
		std::cout << "Data: min = " << minVal << ", max = " << maxVal << ", sum = " << sumVal << std::endl;

		std::cout << "Printing 40th image:" << std::endl;
		for (size_t y = 0; y < numRows; ++y) {
			for (size_t x = 0; x < numCols; ++x) {
				std::cout << charFromFloat(getData(39, y, x));
			}
			std::cout << std::endl;
		}
		std::cout << "Label: " << labels(39) << std::endl;
	}

	// Access a pixel value from the flattened image.
	float getData(uint64_t imgId, uint64_t y, uint64_t x) const {
		return data(imgId, y * numCols + x);
	}

	uint32_t numRows = 0, numCols = 0; //< must be unit32_t to read from file properly!!!
	MlpMatrix data;  // Data matrix (numImages x (numRows*numCols))
	MlpVectori labels;        // Label vector (numImages x 1)
};

class MLP {
public:
	MLP(size_t inputSize, size_t hiddenSize, size_t outputSize, float lr) : lr(lr) {


		srand(42);
		// Initialize weight_1: values in [-1,1] scaled to [-0.01, 0.01]
		weight_1 = MlpMatrix::Random(inputSize, hiddenSize) * 0.01;

		// Initialize bias_1: values in [0,1)
		bias_1 = (MlpRowVectorf::Random(hiddenSize).array() + 1.f) / 2.f;

		// Initialize weight_2: values in [-0.01, 0.01]
		weight_2 = MlpMatrix::Random(hiddenSize, outputSize) * 0.01;

		// Initialize bias_2: values in [0,1)
		bias_2 = (MlpRowVectorf::Random(outputSize).array() + 1.f) / 2.f;

		//std::cout << "weight_1 (first 5 rows):\n" << weight_1.topRows(5) << "\n\n";
		//std::cout << "bias_1:\n" << bias_1 << "\n\n";
		//std::cout << "weight_2 (first 5 rows):\n" << weight_2.topRows(5) << "\n\n";
		//std::cout << "reluWeight_2:\n" << relu(weight_2.topRows(5)) << "\n\n";
		//std::cout << "bias_2:\n" << bias_2 << "\n";

	}

	const MlpMatrix& forward(const Eigen::Ref<const MlpMatrix>& x) {
		// dim(x) is [batchSize x inputSize]
		z1 = (x * weight_1).eval().rowwise() + bias_1;
		a1 = relu(z1);
		z2 = (a1 * weight_2).eval().rowwise() + bias_2;
		a2 = softmax(z2);
		return a2;
	}

	void backward(const Eigen::Ref<const MlpMatrix>& X, const Eigen::Ref<const MlpVectori>& y, const MlpMatrix& output) {
		int m = y.size();      // mini-batch size
		int num_classes = output.cols();

		// 1. One-hot encode the true labels.
		// Create an m x num_classes matrix initialized to zeros.
		y_one_hot = MlpMatrix::Zero(m, num_classes);
		for (int i = 0; i < m; ++i) {
			// Set the column corresponding to the true label to 1.
			y_one_hot(i, y(i)) = 1.0f;
		}

		// 2. Compute gradient at output layer:
		dL_dz2 = output - y_one_hot;

		// 3. Gradients for the second (output) layer:
		dL_dW2 = (a1.transpose() * dL_dz2) / m;
		dL_db2 = dL_dz2.colwise().sum() / m;

		// 4. Backpropagate to the hidden layer:
		dL_da1 = dL_dz2 * weight_2.transpose();
		dL_dz1 = dL_da1.array() * (z1.array() > 0).cast<float>();

		// 5. Gradients for the first (hidden) layer:
		dL_dW1 = (X.transpose() * dL_dz1) / m;
		dL_db1 = dL_dz1.colwise().sum() / m;

		weight_1 -= lr * dL_dW1;
		bias_1 -= lr * dL_db1;
		weight_2 -= lr * dL_dW2;
		bias_2 -= lr * dL_db2;
	}

	void evalEpoch(const Dataset& trainData, int epoch) {
		MlpMatrix full_output = forward(trainData.data);
		double epoch_loss = 0.0;
		for (int i = 0; i < trainData.labels.size(); ++i) {
			// Avoid log(0) by adding a small epsilon if needed.
			float prob = full_output(i, trainData.labels(i));
			epoch_loss += std::log(prob);
		}
		epoch_loss = -epoch_loss / trainData.labels.size();
		losses.push_back(epoch_loss);
		std::cout << (epoch + 1) << "\t" << epoch_loss << std::endl;
	}

	void train(const Dataset& trainData, int epochs = 10, int batch_size = 128) {
		std::cout << "Epoch\tLoss\n";
		int num_samples = trainData.data.rows();
		for (int epoch = 0; epoch < epochs; ++epoch) {
			// Loop over mini-batches.
			for (int i = 0; i < num_samples; i += batch_size) {
				// Determine the actual size of this batch (in case num_samples isn't divisible by batch_size).
				int current_batch_size = std::min(batch_size, num_samples - i);
				// Get a view of the current mini-batch without copying:
				const auto X_batch = trainData.data.block(i, 0, current_batch_size, trainData.data.cols());
				const auto y_batch = trainData.labels.segment(i, current_batch_size);
				// Compute forward pass on the mini-batch.
				const MlpMatrix& output = forward(X_batch);
				// Backpropagate using this mini-batch.
				backward(X_batch, y_batch, output);
			}
			//evalEpoch(trainData, epoch);
		}
	}

	MlpVectori predict(const MlpMatrix& X) {
		// Compute the forward pass to get softmax probabilities.
		const MlpMatrix& output = forward(X);

		// Prepare a vector to hold the predicted class for each row.
		MlpVectori predictions(output.rows());

		// For each sample (row), find the index of the maximum probability.
		for (int i = 0; i < output.rows(); ++i) {
			int maxIndex;
			output.row(i).maxCoeff(&maxIndex);
			predictions(i) = maxIndex;
		}

		return predictions;
	}

	float lr;

	MlpMatrix weight_1; //< dim [inputSize x hiddenSize]
	MlpRowVectorf bias_1;   //< dim [1 x hiddenSize]
	MlpMatrix weight_2; //< dim [hiddenSize x outputSize]
	MlpRowVectorf bias_2;   //< dim [1 x outputSize]

	MlpMatrix z1; //< dim [batchSize x hiddenSize]
	MlpMatrix z2; //< dim [batchSize x outputSize]
	MlpMatrix a1;
	MlpMatrix a2;
	std::vector<float> losses;

	// temp matrices
	MlpMatrix y_one_hot;
	MlpMatrix dL_dz2;
	MlpMatrix dL_dW2;
	MlpMatrix dL_db2;
	MlpMatrix dL_da1;
	MlpMatrix dL_dz1;
	MlpMatrix dL_dW1;
	MlpMatrix dL_db1;
};

int main() {

	Eigen::setNbThreads(32);
	std::cout << "Eigen is using " << Eigen::nbThreads() << " threads.\n";

	Dataset trainData = Dataset("assets.ignored/train-images.idx3-ubyte", "assets.ignored/train-labels.idx1-ubyte");
	Dataset testData = Dataset("assets.ignored/t10k-images.idx3-ubyte", "assets.ignored/t10k-labels.idx1-ubyte");
	//testData.statistics();
	//trainData.statistics();

	size_t inputSize = trainData.numRows * trainData.numCols;
	size_t hiddenSize = 128;
	const auto [_, maxLabel] = std::minmax_element(trainData.labels.begin(), trainData.labels.end());
	size_t outputSize = *maxLabel + 1;
	MLP mlp{ inputSize, hiddenSize, outputSize, 0.01f };

	Time begin = getTime();

	mlp.train(trainData, 80, 64);
	MlpVectori predictions = mlp.predict(testData.data);
	double accuracy = (predictions.array() == testData.labels.array()).cast<double>().mean();
	std::cout << "Test Accuracy: " << accuracy << std::endl;

	Seconds elapsed = getTime() - begin;
	std::cout << "benchmark: " << elapsed;


	return 0;
}
