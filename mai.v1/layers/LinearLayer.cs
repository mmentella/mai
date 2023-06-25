using mai.v1.blas;

namespace mai.v1.layers;

public class LinearLayer
    : ILayer
{
    public int InputSize { get; private set; }
    public int OutputSize { get; private set; }
    public Matrix Weights { get; private set; }
    public Matrix Biases { get; private set; }

    public ILayer? PreviousLayer { get; private set; }
    public ILayer? NextLayer { get; private set; }

    public Matrix Output { get; private set; }

    public LinearLayer(int inputSize, int outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Weights = new Matrix(inputSize, outputSize);
        Biases = new Matrix(1, outputSize);
        Output = default!;
    }

    public void Forward(Matrix input)
    {
        Output = input * Weights + Biases;
        NextLayer?.Forward(Output);
    }

    public void Backward(Matrix input, Matrix outputError, double learningRate)
    {
        Matrix inputError = new(InputSize, OutputSize);
        for (int i = 0; i < OutputSize; i++)
        {
            Biases[i] -= learningRate * outputError[i];
            for (int j = 0; j < InputSize; j++)
            {
                Weights[i, j] -= learningRate * outputError[i] * input[j];
                inputError[j] += outputError[i] * Weights[j * OutputSize + i];
            }
        }
        PreviousLayer?.Backward(input, inputError, learningRate);
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
