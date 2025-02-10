#include <iostream>
#include <DirectXMath.h>
#include <fstream>
#include <vector>
#include <cstdint>
using namespace std;
using namespace DirectX;

// Helper function to swap endianess (if needed)
uint32_t swapEndian(uint32_t val) {
	return ((val >> 24) & 0xff) |
		((val << 8) & 0xff0000) |
		((val >> 8) & 0xff00) |
		((val << 24) & 0xff000000);
}


class MLP {
public:
	MLP() {
		train = Dataset("assets.ignored/train-images.idx3-ubyte", "assets.ignored/train-labels.idx1-ubyte");
		test = Dataset("assets.ignored/t10k-images.idx3-ubyte", "assets.ignored/t10k-labels.idx1-ubyte");

	}

	struct Dataset {
		Dataset() = default;
		Dataset(const string& dataFile, const string& labelFile) {
			cout << "loading " << dataFile << endl;
			ifstream dataIfstream(dataFile, ios::binary);
			if (!dataIfstream) {
				cerr << "Unable to open file!" << endl;
				exit(-1);
			}

			uint32_t magicNumber = 0;

			// Read the header information
			dataIfstream.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
			dataIfstream.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
			dataIfstream.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
			dataIfstream.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

			// MNIST data is stored in big-endian format. If your machine is little-endian,
			// you'll need to swap the byte order.
			magicNumber = swapEndian(magicNumber);
			numImages = swapEndian(numImages);
			numRows = swapEndian(numRows);
			numCols = swapEndian(numCols);

			std::cout << "Magic Number: " << magicNumber << "\n";
			std::cout << "Number of Images: " << numImages << "\n";
			std::cout << "Rows: " << numRows << "\n";
			std::cout << "Columns: " << numCols << "\n";

			data.resize(numImages * numRows * numCols);
			labels.resize(numImages);
			dataIfstream.read(reinterpret_cast<char*>(data.data()), data.size());

			std::streamsize bytesRead = dataIfstream.gcount();
			if (bytesRead != static_cast<std::streamsize>(data.size())) {
				std::cerr << "Error: Only " << bytesRead
					<< " bytes were read, but "
					<< data.size()
					<< " bytes were expected." << std::endl;
				exit(-1);
			}

			//v * Labels
			cout << "loading " << labelFile << endl;
			ifstream labelIfstream(labelFile, ios::binary);
			if (!labelIfstream) {
				cerr << "Unable to open file!" << endl;
				exit(-1);
			}

			// Read the header information
			uint32_t numLabels;
			labelIfstream.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
			labelIfstream.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));

			// MNIST data is stored in big-endian format. If your machine is little-endian,
			// you'll need to swap the byte order.
			magicNumber = swapEndian(magicNumber);
			numLabels = swapEndian(numLabels);

			std::cout << "Magic Number: " << magicNumber << "\n";
			std::cout << "Number of Labels: " << numLabels << "\n";

			if (numLabels != numImages) {
				cerr << "numLabels != numImages\n";
				exit(-1);
			}

			labelIfstream.read(reinterpret_cast<char*>(labels.data()), labels.size());

			bytesRead = labelIfstream.gcount();
			if (bytesRead != static_cast<std::streamsize>(labels.size())) {
				std::cerr << "Error: Only " << bytesRead
					<< " bytes were read, but "
					<< labels.size()
					<< " bytes were expected." << std::endl;
				exit(-1);
			}


		}
		uint32_t numImages = 0, numRows = 0, numCols = 0;
		vector<unsigned char> data;
		vector<unsigned char> labels;
		unsigned char getData(uint64_t imgId, uint64_t y, uint64_t x) const {
			int imgSize = numRows * numCols;
			return data[imgId * imgSize + y * numCols + x];
		}
	};


	Dataset train;
	Dataset test;

};

int main() {
	MLP mlp{};

	return 0;
}
