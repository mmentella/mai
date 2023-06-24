using mai.v1.tensor;

namespace mai.v1.layers;

public class ActivationLayer
    : ILayer
{
    private readonly ActivationFunction activationFunction;

    public ActivationLayer(ActivationFunction activationFunction)
    {
        this.activationFunction = activationFunction;
        Output = default!;
    }

    public ILayer? PreviousLayer { get; private set; }
    public ILayer? NextLayer { get; private set; }

    public Tensor Output { get; private set; }

    public void Backward(Tensor input, Tensor outputError, double learningRate)
    {
        Tensor gradient = activationFunction.Backward(input);
        PreviousLayer?.Backward(gradient, outputError, learningRate);
    }

    public void Forward(Tensor input)
    {
        Tensor activations = activationFunction.Forward(input);
        Output = activations;
        NextLayer?.Forward(activations);
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
