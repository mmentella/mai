using mai.v1.blas;

namespace mai.v1.models;

public class SequentialModel
{
    private readonly List<ILayer> layers = new();

    public Matrix Forward(Matrix input)
    {
        ILayer firstLayer = layers[0];
        ILayer lastLayer = layers[^1];

        firstLayer.Forward(input);

        return lastLayer.Output;
    }

    public void Backward(Matrix input, Matrix outputGradient, double learningRate)
    {
        ILayer lastLayer = layers[^1];
        lastLayer.Backward(input, outputGradient, learningRate);
    }

    public virtual void Add(ILayer layer)
    {
        if (layers.Contains(layer))
        {
            throw new ArgumentException("Layer already exists in model");
        }

        if (layers.Count == 0)
        {
            layers.Add(layer);
            return;
        }

        ILayer lastLayer = layers[^1];
        lastLayer.Stack(layer);

        layers.Add(layer);
    }
}
