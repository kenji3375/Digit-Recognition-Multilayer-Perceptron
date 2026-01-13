#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <iomanip>

#include <fstream>
#include <cstdint>
#include <algorithm>

#include <string>
#include <sstream>

#define E 2.718281828459



inline std::string double_to_string(double value) {
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return oss.str();
}



std::vector<double> softmaxLayer(std::vector<double> output) {
    std::vector<double> probabilities(output.size());

    double maxv = *std::max_element(output.begin(), output.end());

    for(double &v : output) v -= maxv;

    double denominator{};

    for (int i=0; i<output.size(); ++i) {
        denominator += std::exp(output[i]);
    }

    for (int i=0; i<output.size(); ++i) {
        probabilities[i] = std::exp(output[i])/denominator;
    }

    return probabilities;
}

inline double relu(double n) {
    return std::max(0.0, n);
}

inline double noActivation(double x) {
    return x;
}


inline double randomDouble(double min, double max, int p) {             // p - precision
    static thread_local std::mt19937_64 rng(std::random_device{}());
    
    // scale factor for the desired precision
    double scale = std::pow(10.0, p);

    // int distribution for scaled range
    std::uniform_int_distribution<long long> dist(
        static_cast<long long>(std::round(min * scale)),
        static_cast<long long>(std::round(max * scale))
    );

    // convert back to double with p-decimal precision
    return dist(rng) / scale;
}

struct Neuron {
    double bias;
    std::vector<double> weights;

    double (*activation)(double);   //activation function


    
    Neuron(int weightsNumber, double (*act)(double)=&relu) {
        double limit = std::sqrt(2.0/weightsNumber);
        weights = std::vector<double>(weightsNumber, randomDouble(-limit,limit,4));
        bias    = randomDouble(-limit,limit,2);

        activation = act;
    }

    double activate(std::vector<double> layer) {    //activate method produces neuron's expected output based on given input
        double sum = 0;
        for(int i=0; i<layer.size(); ++i) {
            sum += layer[i] * weights[i];
        }
        sum += bias;
        return activation(sum);
    }
    
    // lastSum and lastActivation are going to be used in the backpropagation (training) process
    double lastSum;
    double lastActivation;

    double activateTraining(std::vector<double> layer) {
        double sum = 0;
        for(int i=0; i<layer.size(); ++i) {
            sum += layer[i] * weights[i];
        }
        sum += bias;

        lastSum = sum;
        lastActivation = activation(sum);
        return lastActivation;
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
        }

        neurons = std::vector<std::vector<Neuron>>(layers);

        //first layer is without actual neurons, so it's not set to anything

        for(int l=1; l<layers-1; ++l) {
            neurons[l] = std::vector<Neuron>(layerSize[l], Neuron(layerSize[l-1]));
            for(int n=0; n<layerSize[l]; ++n) {
                neurons[l][n] = Neuron(layerSize[l-1]);
            }
        }
        neurons[layers-1] = std::vector<Neuron>(layerSize[layers-1], Neuron(layerSize[layers-2]));
        for(int n=0; n<layerSize[layers-1]; ++n) {
            neurons[layers-1][n] = Neuron(layerSize[layers-2], noActivation); //last layer's output will be passed through softmax activatin function, which is dependent on other neurons, so neurons on the individual level have no activation set
        }


    }

    std::vector<double> forwardPass(std::vector<double> input) {    //computes output of the NN

        if(input.size() == layerSize[0]) {

            std::vector<double> tempLayer;

            for(int L=1; L<layers; ++L) {
                tempLayer = std::vector<double>(layerSize[L]);
                for(int n=0; n<layerSize[L]; ++n) {
                    tempLayer[n] = neurons[L][n].activateTraining(input);
                }
                input = tempLayer;
            }
            // return tempLayer;
            return softmaxLayer(tempLayer);

        } else {
            return std::vector<double>(1,-1);
        }
    }

    

    double loss(std::vector<double> input, std::vector<double> expected) {  //  MSE loss function (MSE - mean squared error)
        double loss{};
        std::vector<double> res;
        res = this->forwardPass(input);

        for(int i=0; i<res.size(); ++i) {
            loss += std::pow(res[i]-expected[i],2)/2;
        }

        return loss;
    }

    void backpropagate(std::vector<double> input, std::vector<double> expected, double learningRate, bool showdata=false) {
        std::vector<double> output = forwardPass(input); //computing output to set all neurons' correct lastSum and lastActivation

        std::vector<std::vector<double>> deltas(layers); //deltas will be used to compute gradient, gradient will be used to subtract correct values from weights and biases

        for(int i=0; i<layers; ++i) {
            deltas[i] = std::vector<double>(layerSize[i], 0);
        }

        for(int i=0; i<output.size(); ++i) {    // because softmax is used instead of ReLU, this layer will be treated differently
            deltas[layers-1][i] = output[i] - expected[i];  // in case of sotfmax activation this is how delta is computed
        }
        
        // deltas for hidden layers
        double sum;
        for(int L=layers-2; L>0; --L) {             //following the formula for deltas of neurons using relu
            for(int i=0; i<layerSize[L]; ++i) {
                sum=0;
                for(int k=0; k<layerSize[L+1]; ++k) {
                    sum += neurons[L+1][k].weights[i] * deltas[L+1][k];
                }
                sum *= (neurons[L][i].lastSum>0 ? 1:0);     // if neuron wasn't activated, the delta should be 0
                deltas[L][i] = sum;
            }
        }

        //gradient will not be calculated separately
        
        for(int n=0; n<layerSize[1]; ++n) {
            for(int w=0; w<layerSize[0]; ++w) {
                neurons[1][n].weights[w] -= learningRate * deltas[1][n] * input[w];
            }
            neurons[1][n].bias -= learningRate * deltas[1][n];
        }
        for(int L=2; L<layers; ++L) {
            for(int n=0; n<layerSize[L]; ++n) {
                for(int w=0; w<layerSize[L-1]; ++w) {
                    neurons[L][n].weights[w] -= learningRate * deltas[L][n] * neurons[L-1][w].lastActivation; //subtracting weight gradient
                }
                neurons[L][n].bias -= learningRate * deltas[L][n];  //subtracting bias gradient
            }
        }
    }

    void trainBatch(std::vector<std::vector<std::vector<double>>> batch, double learningRate) {

        double size = batch.size();

        double loss_sum{};

        for(int epoch=0; epoch<size; ++epoch) {
            
            if(! (epoch%100)) {     //every 100th epoch clear the line and print status

                std::cout << "\033[1G";
                std::cout<<"epoch: "<<epoch<<". average loss: "<<(loss_sum/epoch);//<<std::endl;
            }

            loss_sum += loss(batch[epoch][0],batch[epoch][1]);

            backpropagate(batch[epoch][0], batch[epoch][1], learningRate);
        }
        std::cout<<"\n";
    }

    void saveNN() {         //save neural network into output.txt file
        std::string code = "std::vector<std::vector<std::vector<double>>> weights = {\n";
        for(int L=1; L<layers; ++L) {
            code += "   {\n";
            for(int n=0; n<layerSize[L]; ++n) {
                code += "       {\n";
                for(int w=0; w<neurons[L][n].weights.size(); ++w) {
                    code += "           "+ double_to_string(neurons[L][n].weights[w]);
                    code += ", \n";
                }
                code += "       },\n";

            }
            code += "   },";
        }
        code += "};\n\n";
        
        code += "std::vector<std::vector<double>> biases = {\n";
        for(int L=1; L<layers; ++L) {
            code += "   {\n";
            for(int n=0; n<layerSize[L]; ++n) {
                code += "       " + double_to_string(neurons[L][n].bias) + ",\n";
                // code += "       " + std::format("{:.2f}", neurons[L][n].bias) + ",\n";
            }
            code += "   },\n";
        }
        code += "};\n";
        
        
        std::ofstream file("output.txt");
        file << code;
        file.close();
    }
    
};



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
    std::vector<int> labels;

    int size;
    
    NeuralData(std::string img, std::string nameLabels) {       //reading of training/test idx data from file "img"

        img = "./mnist_digits/"+img;
        
        uint8_t byte{};
        uint32_t bytes4{};
        uint32_t dim{};
        int dims{};
        //read images
        std::ifstream imgfile(img, std::ios::binary);

        imgfile.read(reinterpret_cast<char*>(&bytes4),4); // "magic number"
        
        imgfile.read(reinterpret_cast<char*>(&dim),4);    // first dimension (which means amount of images)
        
        dim = _byteswap_uint32(static_cast<uint64_t>(dim));

        size = dim;
        
        
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
        
        lblfile.read(reinterpret_cast<char*>(&bytes4),4); // "magic number"
        
        lblfile.read(reinterpret_cast<char*>(&dim),4);    // first (and only) dimension
        
        labels = std::vector<int>(dim,0);
        
        
        for(int i=0; i<dim; ++i) {
            lblfile.read(reinterpret_cast<char*>(&byte),1);
            labels[i] = byte;
        }
        
        lblfile.close();

        std::cout<<"Loaded "<<size<<" images.\n";
        
    }

    void display(int index) {   // display the read image data as ascii art (to test the reading of data)
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



    std::vector<double> getX(int i) {   //returns single X (input) for NN
        std::vector<double> X;
        X = std::vector<double>(28*28, 0);
        for(int y=0; y<28; ++y) {
            for(int x=0; x<28; ++x) {
                X[y*28+x] = images[i][y][x];
            }
        }
        return X;
    }

    std::vector<double> getY(int i) {   //returns single Y (output) for NN
        std::vector<double> Y(10, 0);
        Y[labels[i]] = 1;
        return Y;
    }

    std::vector<std::vector<std::vector<double>>> getBatch(int istart, int len) {   //returns neural data from specified range
        std::vector<std::vector<std::vector<double>>> batch(len);
        for(int i=0; i<len; ++i) {
            std::vector<std::vector<double>> XY(2);
            XY[0] = getX(i+istart);
            XY[1] = getY(i+istart);
            
            batch[i] = XY;
        }
        
        return batch;
    }

};






void testAccuracy(MultiLayerPerceptron nn, NeuralData data) {   //tests accuracy over const n examples for given NN and data
    const int n = 10000;

    int accuratePredictions{};

    std::vector<double> output(10);

    int maxi;

    for(int i=0; i<n; ++i) {
        output = nn.forwardPass(data.getX(i));
        maxi=0;
        for(int x=0; x<10; ++x) {
            if(output[x]>output[maxi]) {
                maxi = x;
            }
        }
        if(maxi == data.labels[i]) {
            ++accuratePredictions;
        }
    }
    std::cout<<"accurate predictions: "<<static_cast<double>(accuratePredictions)/static_cast<double>(n*100)<<"%"<<std::endl;
}

double av_loss(MultiLayerPerceptron nn, NeuralData data) {  //returns average loss for const n examples
    double losssum{};
    const int n = 250;
    for(int i=0; i<n; ++i) {
        losssum += nn.loss(data.getX(i), data.getY(i));
    }
    return losssum/static_cast<double>(n);
}





int main(void)
{
    std::string labelsName = "train-labels.idx1-ubyte";
    std::string imagesName = "train-images.idx3-ubyte";
    
    NeuralData data(imagesName, labelsName);        //loading training data
    
    labelsName = "t10k-labels.idx1-ubyte";
    imagesName = "t10k-images.idx3-ubyte";
    
    NeuralData data_test(imagesName, labelsName);   //loading test data
    
    //setting MLP's dimensions (28*28 input layer, 1 hidden layer with 128 neurons, one output layer with 10 neurons)
    std::vector<int> layers = {28*28, 128, 10};
    
    MultiLayerPerceptron nn(layers);    //creation of MLP with given layers
    
    testAccuracy(nn,data_test);         //test accuracy before training
    
    // i am aware of the risks of overfitting and training on test data
    // i'll take available data i can to maximise results in actual program
    nn.trainBatch(data.getBatch(0,60000), 0.01);
    nn.trainBatch(data_test.getBatch(0,10000), 0.003375);
    nn.trainBatch(data.getBatch(20000,40000), 0.002);
    nn.trainBatch(data_test.getBatch(0,10000), 0.001);
    
    testAccuracy(nn, data_test);       //test accuracy after training
    
    nn.saveNN();                       //save nn to file output.txt
    
    return 0;
}
