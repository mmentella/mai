namespace mai.v1.layers;

public class LinearLayer
    : ILayer
{
    public int InputSize { get; private set; }
    public int OutputSize { get; private set; }
    public double[] Weights { get; private set; }
    public double[] Biases { get; private set; }

    public ILayer? PreviousLayer { get; private set; }
    public ILayer? NextLayer { get; private set; }

    public double[] Output { get; private set; }

    public LinearLayer(int inputSize, int outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Weights = new double[inputSize * outputSize];
        Biases = new double[outputSize];
        Output = Array.Empty<double>();

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

    public void Forward(double[] input)
    {
        double[] output = new double[OutputSize];
        for (int i = 0; i < OutputSize; i++)
        {
            output[i] = Biases[i];
            for (int j = 0; j < InputSize; j++)
            {
                output[i] += input[j] * Weights[j * OutputSize + i];
            }
        }
        Output = output;
        NextLayer?.Forward(output);
    }

    public void Backward(double[] input, double[] outputError, double learningRate)
    {
        double[] inputError = new double[InputSize];
        for (int i = 0; i < OutputSize; i++)
        {
            Biases[i] -= learningRate * outputError[i];
            for (int j = 0; j < InputSize; j++)
            {
                Weights[j * OutputSize + i] -= learningRate * outputError[i] * input[j];
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
