﻿using mai.v1.activation;
using mai.v1.blas;
using mai.v1.loss;

namespace mai.v1.layers;

public class CrossEntropySoftmaxLayer
    : ILayer
{
    private Matrix input;
    private Matrix output;
    private Matrix weightedInput;
    private ActivationFunction activationFunction;
    private LossFunction lossFunction;

    public CrossEntropySoftmaxLayer(int inputSize, int outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;

        Weights = new Matrix(inputSize, outputSize);

        activationFunction = new SoftmaxActivationFunction();
        lossFunction = new CrossEntropyLossFunction();

        input = default!;
        output = default!;
        weightedInput = default!;
    }

    public int InputSize { get; private set; }
    public int OutputSize { get; private set; }
    public Matrix Weights { get; private set; }

    public Matrix Output => output;

    public ILayer? PreviousLayer { get; private set; }
    public ILayer? NextLayer { get; private set; }

    public void Backward(Matrix input, Matrix outputError, double learningRate)
    {
        Matrix dedwi = output - input;
        Weights -= learningRate * (input * dedwi);
    }

    public void Forward(Matrix input)
    {
        this.input = input;
        weightedInput = input * Weights;
        output = activationFunction.Forward(weightedInput);
    }

    public void SetNextLayer(ILayer layer)
    {
        NextLayer = layer;
    }

    public void SetPreviousLayer(ILayer layer)
    {
        PreviousLayer = layer;
    }

    public void Stack(ILayer layer)
    {
        NextLayer = layer;
        layer.SetPreviousLayer(this);
    }
}
