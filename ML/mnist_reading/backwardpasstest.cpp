#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <iomanip>

#include <fstream>
#include <cstdint>

#define E   2.71828182846




inline uint32_t _byteswap_uint32 (uint32_t n) {
    return 
    ((n & 0xFF000000) >> 24) |
    ((n & 0x00FF0000) >>  8) |
    ((n & 0x0000FF00) <<  8) |
    ((n & 0x000000FF) << 24);
}

struct NeuralData
{
    std::vector<std::vector<std::vector<double>>> images;
    // std::vector<std::vector<std::vector<uint8_t>>> images_uint8_t;
    std::vector<int> labels;

    NeuralData(std::string img, std::string nameLabels) {

        
        uint8_t byte{};
        uint32_t bytes4{};
        uint32_t dim{};
        int dims{};
        //read images
        std::ifstream imgfile(img, std::ios::binary);

        if(imgfile)std::cout<<"found images file.\n";
        imgfile.read(reinterpret_cast<char*>(&bytes4),4); // "magic number"
        
        imgfile.read(reinterpret_cast<char*>(&dim),4);    // first dimension (which means amount of images)
        
        dim = _byteswap_uint32(static_cast<uint64_t>(dim));

        std::cout<<dim<<"\n";
        
        
        images = std::vector<std::vector<std::vector<double>>>(
            dim, std::vector<std::vector<double>>(28,std::vector<double>(28, 0))
        );

        imgfile.read(reinterpret_cast<char*>(&bytes4),4); //other 2 dimensions (size of image) but they're always 28x28
        imgfile.read(reinterpret_cast<char*>(&bytes4),4);
        
        for(int i=0; i<dim; ++i) {
            for(int y=0; y<28; ++y) {
                for(int x=0; x<28; ++x) {
                    imgfile.read(reinterpret_cast<char*>(&byte),1);
                    //255 - 1
                    //0   - 0
                    images[i][y][x] = (double)byte/255.0;
                }
            }
        }
        
        imgfile.close();

        //read labels
        
        std::ifstream lblfile(nameLabels, std::ios::binary);
        
        if(lblfile)std::cout<<"found labels file.\n";
        
        lblfile.read(reinterpret_cast<char*>(&bytes4),4); // "magic number"
        
        lblfile.read(reinterpret_cast<char*>(&dim),4);    // first (and only) dimension
        
        labels = std::vector<int>(dim,0);
    
        
        for(int i=0; i<dim; ++i) {
            lblfile.read(reinterpret_cast<char*>(&byte),1);
            labels[i] = byte;
        }
        
        lblfile.close();
        
    }

    void display(int index) {
        std::cout<<"\n";
        std::cout<<labels[index];
        std::cout<<"\n";
        for(int y=0; y<28; ++y) {
            for(int x=0; x<28; ++x) {
                if(images[index][y][x]) {
                    if(images[index][y][x] > 0.75) {
                        std::cout<<"##";
                    } else if(images[index][y][x] > 0.5) {
                        std::cout<<"++";
                    } else if(images[index][y][x] > 0.25) {
                        std::cout<<"--";
                    } else {
                        std::cout<<"..";
                    }
                } else std::cout<<"  ";
            } std::cout<<"\n";
        }
        std::cout<<"\n";
        std::cout<<"\n";
    }

    std::vector<double> expected(int index) {
        std::vector<double> exp(10,0);
        exp[labels[index]] = 1;

        return exp;
    }

    std::vector<double> getInputLayer(int index) {
        std::vector<double> res(28*28);
        for(int y=0; y<28; ++y) {
            for(int x=0; x<28; ++x) {
                res[28*y+x] = images[index][y][x];
            }
        }

        return res;
    }
};





inline double relu(double n) {
    return std::max(0.0, n);
}

inline double fastSigmoid(double x) {
    // return (x/(1+abs(x)))/2 + 0.5;      // not exactly sigmod but is functionally the same
    return (1/(1+std::pow(E, -x)));      // not exactly sigmod but is functionally the same
}

inline double randomDouble(double min, double max, int p) {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    
    // Scale factor for the desired precision
    double scale = std::pow(10.0, p);

    // Integer distribution for scaled range
    std::uniform_int_distribution<long long> dist(
        static_cast<long long>(std::round(min * scale)),
        static_cast<long long>(std::round(max * scale))
    );

    // Convert back to double with p-decimal precision
    return dist(rng) / scale;
}

struct Neuron {
    double bias;
    std::vector<double> weights;

    double (*activation)(double);


    
    Neuron(int weightsNumber, double (*act)(double)=&relu) {
        weights = std::vector<double>(weightsNumber, randomDouble(-2,2,2));
        bias    = randomDouble(-2,2,2);

        activation = act;
    }

    double activate(std::vector<double> layer) {
        double sum = 0;
        for(int i=0; i<layer.size(); ++i) {
            sum += layer[i] * weights[i];
        }
        sum += bias;
        return activation(sum);
    }
};



struct MultiLayerPerceptron {
    std::vector<std::vector<Neuron>> neurons;
    int layers;
    std::vector<int> layerSize;
    
    MultiLayerPerceptron(std::vector<int> matrix) {
        this->layers = matrix.size();
        layerSize = std::vector<int>(layers);
        for(int i=0; i<layers; ++i) {
            layerSize[i] = matrix[i];
            // layerSize.emplace_back(matrix[i]);
        }

        neurons = std::vector<std::vector<Neuron>>(layers);

        // neurons[0] = std::vector<Neuron>(layerSize[0], Neuron(1,1,1)); //first layer will be never used anyway

        for(int l=1; l<layers-1; ++l) {
            neurons[l] = std::vector<Neuron>(layerSize[l], Neuron(layerSize[l-1]));
            for(int n=0; n<layerSize[l]; ++n) {
                neurons[l][n] = Neuron(layerSize[l-1]);
            }
        }
        neurons[layers-1] = std::vector<Neuron>(layerSize[layers-1], Neuron(layerSize[layers-2]));
        for(int n=0; n<layerSize[layers-1]; ++n) {
            neurons[layers-1][n] = Neuron(layerSize[layers-2], fastSigmoid);
        }
    }

    std::vector<double> out(std::vector<double> input) {

        if(input.size() == layerSize[0]) {

            std::vector<double> tempLayer;

            for(int L=1; L<layers; ++L) {
                tempLayer = std::vector<double>(layerSize[L]);
                for(int n=0; n<layerSize[L]; ++n) {
                    tempLayer[n] = neurons[L][n].activate(input);
                }
                input = tempLayer;
            }
            return tempLayer;

        } else {
            return std::vector<double>(1,0);
        }
    }

    double loss(std::vector<double> input, std::vector<double> expected) {
        double loss{};
        std::vector<double> res;
        res = this->out(input);

        for(int i=0; i<res.size(); ++i) {
            loss += res[i]*res[i];
        }

        return loss;
    }
};


// void test() {
//     std::vector<int> layers = {4,4,2};

//     MultiLayerPerceptron nn(layers);

//     std::vector<double> input = {2,6,2,5};

//     std::vector<double> output = nn.out(input);

//     for(int i=0; i<output.size(); ++i) {
//         std::cout<<std::fixed<<std::setw(5)<<output[i]<<"\n";
//     }
// }
void test() {
    std::vector<int> layers = {4,7,5};

    MultiLayerPerceptron nn(layers);

    std::vector<double> input = {2,6,2,5};
    
    std::vector<double> output = nn.out(input);
    
    for(int i=0; i<output.size(); ++i) {
        std::cout<<std::fixed<<std::setw(5)<<output[i]<<"\n";
    }
    std::vector<double> expected = {1, 0, 0.5, 0, 0, 1, 1};

    double loss = nn.loss(input, expected);
    std::cout<<"loss: "<<loss<<"\n";
}

void test2() {
    std::string labelsName = "t10k-labels.idx1-ubyte";
    std::string imagesName = "t10k-images.idx3-ubyte";

    NeuralData data(imagesName, labelsName);

    // data.display(0);
    // data.display(1);


    std::cout<<data.labels[0]<<"\n";

    std::vector<int> layers = {28*28, 128, 128, 10};

    MultiLayerPerceptron nn(layers);

    // std::vector<double> input = {2,6,2,5};
    std::vector<double> input = data.getInputLayer(0);
    
    
    std::vector<double> output = nn.out(input);
    
    for(int i=0; i<output.size(); ++i) {
        std::cout<<std::fixed<<std::setw(5)<<output[i]<<"\n";
    }
    std::vector<double> expected = data.expected(0);

    double loss = nn.loss(input, expected);
    std::cout<<"loss: "<<loss<<"\n";


}


int main(int argc, char const *argv[])
{
    test2();
    // std::cout<<"\n\n";
    // test2();
    // std::cout<<"\n\n";
    // test2();
    // std::cout<<"\n\n";
    return 0;
}
