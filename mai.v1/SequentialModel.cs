using mai.v1.tensor;

namespace mai.v1;

public class SequentialModel
{
    private readonly List<ILayer> layers = new();

    public Tensor Forward(Tensor input)
    {
        ILayer firstLayer = layers[0];
        ILayer lastLayer = layers[^1];

        firstLayer.Forward(input);
        
        return lastLayer.Output;
    }

    public void Backward(Tensor input, Tensor outputGradient, double learningRate)
    {
        ILayer lastLayer = layers[^1];
        lastLayer.Backward(input, outputGradient, learningRate);
    }

    public virtual void Add(ILayer layer)
    {
        if(layers.Contains(layer))
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
