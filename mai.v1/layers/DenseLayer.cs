namespace mai.v1.layers;

internal class DenseLayer
    : ILayer
{
    public DenseLayer(int inputSize, int outputSize, ActivationFunction activationFunction)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Weights = new double[inputSize * outputSize];
        Biases = new double[outputSize];

        linearLayer = new LinearLayer(inputSize, outputSize);
        activationLayer = new ActivationLayer(activationFunction);

        linearLayer.Stack(activationLayer);

        Output = activationLayer.Output;
    }

    public int InputSize { get; private set; }
    public int OutputSize { get; private set; }
    public double[] Weights { get; private set; }
    public double[] Biases { get; private set; }

    private readonly LinearLayer linearLayer;
    private readonly ActivationLayer activationLayer;

    public double[] Output { get; private set; }

    public ILayer? PreviousLayer { get; private set; }
    public ILayer? NextLayer { get; private set; }

    public void Backward(double[] input, double[] outputError, double learningRate)
    {
        activationLayer.Backward(input, outputError, learningRate);
    }

    public void Forward(double[] input)
    {
        linearLayer.Forward(input);
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
