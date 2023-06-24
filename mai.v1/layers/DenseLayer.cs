using mai.v1.tensor;

namespace mai.v1.layers;

public class DenseLayer
    : ILayer
{
    public DenseLayer(int inputSize, int outputSize, ActivationFunction activationFunction)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Weights = new Tensor(inputSize, outputSize);
        Biases = new Tensor(outputSize);

        linearLayer = new LinearLayer(inputSize, outputSize);
        activationLayer = new ActivationLayer(activationFunction);

        linearLayer.Stack(activationLayer);

        Output = activationLayer.Output;
    }

    public int InputSize { get; private set; }
    public int OutputSize { get; private set; }
    public Tensor Weights { get; private set; }
    public Tensor Biases { get; private set; }

    private readonly LinearLayer linearLayer;
    private readonly ActivationLayer activationLayer;

    public Tensor Output { get; private set; }

    public ILayer? PreviousLayer { get; private set; }
    public ILayer? NextLayer { get; private set; }

    public void Backward(Tensor input, Tensor outputError, double learningRate)
    {
        activationLayer.Backward(input, outputError, learningRate);
    }

    public void Forward(Tensor input)
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
