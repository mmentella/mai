using mai.v1.blas;

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

    public Matrix Output { get; private set; }

    public void Backward(Matrix input, Matrix outputError, double learningRate)
    {
        Matrix gradient = activationFunction.Backward(outputError);
        PreviousLayer?.Backward(input, gradient, learningRate);
    }

    public void Forward(Matrix input)
    {
        Matrix activations = activationFunction.Forward(input);
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
