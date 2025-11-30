#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

#define E 2.718281828459

inline double relu(double n);
inline double noActivation(double x);

std::vector<double> softmaxLayer(std::vector<double> output);
inline double randomDouble(double min, double max, int precision);

struct Neuron {
    double bias;
    std::vector<double> weights;
    double (*activation)(double);

    double lastSum;
    double lastActivation;

    Neuron(int weightsNumber, double (*act)(double));
    double activate(std::vector<double> layer);
};

struct MultiLayerPerceptron {
    std::vector<std::vector<Neuron>> neurons;
    int layers;
    std::vector<int> layerSize;

    MultiLayerPerceptron() : layers(0) {}
    MultiLayerPerceptron(std::vector<int> matrix);

    std::vector<double> forwardPass(std::vector<double> input);
    double loss(std::vector<double> input, std::vector<double> expected);
    void backpropagate(std::vector<double> input, std::vector<double> expected, double learningRate, bool showdata=false);
    void trainBatch(std::vector<std::vector<std::vector<double>>> batch, double learningRate);
};
