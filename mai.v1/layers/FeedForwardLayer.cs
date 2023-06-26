using mai.v1.blas;

namespace mai.v1.layers;

public class FeedForwardLayer
    : ILayer
{
    public int InputSize { get; private set; }
    public int OutputSize { get; private set; }
    public Matrix Weights { get; private set; }

    public ILayer? PreviousLayer { get; private set; }
    public ILayer? NextLayer { get; private set; }

    public Matrix Output { get; private set; }

    public FeedForwardLayer(int inputSize, int outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Weights = new Matrix(inputSize, outputSize);
        Output = default!;
    }

    public void Forward(Matrix input)
    {
        Output = input * Weights;
        NextLayer?.Forward(Output);
    }

    public void Backward(Matrix input, Matrix outputError, double learningRate)
    {
        Weights -= learningRate * (input * outputError);

        PreviousLayer?.Backward(input, outputError * Weights.Transpose(), learningRate);
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
