#include "nnlib.h"
#include "network.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

#define E 2.718281828459

inline double relu(double n) {
    return std::max(0.0, n);
}

inline double noActivation(double x) {
    return x;
}

inline double randomDouble(double min, double max, int p) {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    
    double scale = std::pow(10.0, p);
    std::uniform_int_distribution<long long> dist(
        static_cast<long long>(std::round(min * scale)),
        static_cast<long long>(std::round(max * scale))
    );

    return dist(rng) / scale;
}

std::vector<double> softmaxLayer(std::vector<double> output) {
    std::vector<double> probabilities(output.size());
    double maxv = *std::max_element(output.begin(), output.end());
    for(double &v : output) v -= maxv;

    double denominator = 0.0;
    for (double &v : output) denominator += std::exp(v);

    for (int i = 0; i < output.size(); ++i)
        probabilities[i] = std::exp(output[i]) / denominator;

    return probabilities;
}

Neuron::Neuron(int weightsNumber, double (*act)(double)=&relu) {
    double limit = std::sqrt(2.0 / weightsNumber);
    weights = std::vector<double>(weightsNumber, randomDouble(-limit, limit, 4));
    bias = randomDouble(-2, 2, 2);
    activation = act;
}

double Neuron::activate(std::vector<double> layer) {
    double sum = 0;
    for(int i=0; i<layer.size(); ++i)
        sum += layer[i] * weights[i];
    sum += bias;
    return activation(sum);
}

MultiLayerPerceptron::MultiLayerPerceptron(std::vector<int> matrix) {
    layers = matrix.size();
    layerSize = std::vector<int>(layers);
    for(int i=0; i<layers; ++i)
        layerSize[i] = matrix[i];

    neurons = std::vector<std::vector<Neuron>>(layers);

    for(int l=1; l<layers-1; ++l) {
        neurons[l] = std::vector<Neuron>(layerSize[l], Neuron(layerSize[l-1]));
        for(int n=0; n<layerSize[l]; ++n)
            neurons[l][n] = Neuron(layerSize[l-1]);
    }

    neurons[layers-1] = std::vector<Neuron>(layerSize[layers-1], Neuron(layerSize[layers-2]));
    for(int n=0; n<layerSize[layers-1]; ++n)
        neurons[layers-1][n] = Neuron(layerSize[layers-2], noActivation);

    
    for(int L=1; L<layers; ++L) {   //loading weights from network.cpp
        for(int n=0; n<layerSize[L]; ++n) {
            for(int w=0; w<layerSize[L-1]; ++w) {
                neurons[L][n].weights[w] = weights[L-1][n][w];
                neurons[L][n].bias = biases[L-1][n];
            }
        }
    }
}

std::vector<double> MultiLayerPerceptron::forwardPass(std::vector<double> input) {
    if(input.size() != layerSize[0])
        return std::vector<double>(1,-1);

    std::vector<double> tempLayer;
    for(int L=1; L<layers; ++L) {
        tempLayer = std::vector<double>(layerSize[L]);
        for(int n=0; n<layerSize[L]; ++n)
            tempLayer[n] = neurons[L][n].activate(input);
        input = tempLayer;
    }

    return softmaxLayer(tempLayer);
}

double MultiLayerPerceptron::loss(std::vector<double> input, std::vector<double> expected) {
    std::vector<double> res = forwardPass(input);
    double loss = 0;
    for(int i=0; i<res.size(); ++i)
        loss += std::pow(res[i]-expected[i],2)/2;
    return loss;
}

void MultiLayerPerceptron::backpropagate(std::vector<double> input, std::vector<double> expected, double learningRate, bool showdata) {
    std::vector<double> output = forwardPass(input);
    std::vector<std::vector<double>> deltas(layers);
    for(int i=0; i<layers; ++i)
        deltas[i] = std::vector<double>(layerSize[i], 0);

    for(int i=0; i<output.size(); ++i)
        deltas[layers-1][i] = output[i]-expected[i];

    double sum;
    for(int L=layers-2; L>0; --L) {
        for(int i=0; i<layerSize[L]; ++i) {
            sum = 0;
            for(int k=0; k<layerSize[L+1]; ++k)
                sum += neurons[L+1][k].weights[i]*deltas[L+1][k];
            sum *= (neurons[L][i].lastSum>0?1:0);
            deltas[L][i] = sum;
        }
    }

    for(int n=0; n<layerSize[1]; ++n) {
        for(int w=0; w<layerSize[0]; ++w)
            neurons[1][n].weights[w] -= learningRate*deltas[1][n]*input[w];
        neurons[1][n].bias -= learningRate*deltas[1][n];
    }

    for(int L=2; L<layers; ++L) {
        for(int n=0; n<layerSize[L]; ++n) {
            for(int w=0; w<layerSize[L-1]; ++w)
                neurons[L][n].weights[w] -= learningRate*deltas[L][n]*neurons[L-1][w].lastActivation;
            neurons[L][n].bias -= learningRate*deltas[L][n];
        }
    }
}

void MultiLayerPerceptron::trainBatch(std::vector<std::vector<std::vector<double>>> batch, double learningRate) {
    std::cout<<"started training.\n";
    double size = batch.size();
    for(int epoch=0; epoch<size; ++epoch) {
        std::cout<<"epoch: "<<epoch<<std::endl;
        backpropagate(batch[epoch][0], batch[epoch][1], learningRate);
    }
    std::cout<<"Training on batch finished. \n";
}
