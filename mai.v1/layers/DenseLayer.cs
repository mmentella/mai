using mai.v1.blas;

namespace mai.v1.layers;

public class DenseLayer
    : ILayer
{
    public DenseLayer(int inputSize, int outputSize, ActivationFunction activationFunction)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Weights = new Matrix(inputSize, outputSize);
        Biases = new Matrix(1, outputSize);

        linearLayer = new LinearLayer(inputSize, outputSize);
        activationLayer = new ActivationLayer(activationFunction);

        linearLayer.Stack(activationLayer);

        Output = activationLayer.Output;
    }

    public int InputSize { get; private set; }
    public int OutputSize { get; private set; }
    public Matrix Weights { get; private set; }
    public Matrix Biases { get; private set; }

    private readonly LinearLayer linearLayer;
    private readonly ActivationLayer activationLayer;

    public Matrix Output { get; private set; }

    public ILayer? PreviousLayer { get; private set; }
    public ILayer? NextLayer { get; private set; }

    public void Backward(Matrix input, Matrix outputError, double learningRate)
    {
        activationLayer.Backward(input, outputError, learningRate);
    }

    public void Forward(Matrix input)
    {
        linearLayer.Forward(input);
        Output = activationLayer.Output;
    }

    public void SetNextLayer(ILayer layer)
    {
        activationLayer.SetNextLayer(layer);
    }

    public void SetPreviousLayer(ILayer layer)
    {
        linearLayer.SetPreviousLayer(layer);
    }

    public void Stack(ILayer layer)
    {
        activationLayer.Stack(layer);
    }
}
