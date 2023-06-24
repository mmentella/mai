using mai.v1.tensor;

namespace mai.v1.layers;

public class LinearLayer
    : ILayer
{
    public int InputSize { get; private set; }
    public int OutputSize { get; private set; }
    public Tensor Weights { get; private set; }
    public Tensor Biases { get; private set; }

    public ILayer? PreviousLayer { get; private set; }
    public ILayer? NextLayer { get; private set; }

    public Tensor Output { get; private set; }

    public LinearLayer(int inputSize, int outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Weights = new Tensor(inputSize, outputSize);
        Biases = new Tensor(outputSize);
        Output = default!;

        Random random = new();
        for (int i = 0; i < OutputSize; i++)
        {
            Biases[i] = 0.001 * random.NextDouble();
            for (int j = 0; j < InputSize; j++)
            {
                Weights[j * OutputSize + i] = 0.001 * random.NextDouble();
            }
        }
    }

    public void Forward(Tensor input)
    {
        Tensor output = new(OutputSize);
        for (int i = 0; i < OutputSize; i++)
        {
            output[i] = Biases[i];
            for (int j = 0; j < InputSize; j++)
            {
                output[i] += input[j] * Weights[i, j];
            }
        }
        Output = output;
        NextLayer?.Forward(output);
    }

    public void Backward(Tensor input, Tensor outputError, double learningRate)
    {
        Tensor inputError = new(InputSize);
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
