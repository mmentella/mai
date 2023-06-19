using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mai.v1.layers;

public class ActivationLayer
    : ILayer
{
    private readonly ActivationFunction activationFunction;

    public ActivationLayer(ActivationFunction activationFunction)
    {
        this.activationFunction = activationFunction;
    }

    public ILayer? PreviousLayer { get; private set; }
    public ILayer? NextLayer { get; private set; }

    public double[] Output { get; private set; }

    public void Backward(double[] input, double[] outputError, double learningRate)
    {
        double[] gradient = activationFunction.Backward(input);
        PreviousLayer?.Backward(gradient, outputError, learningRate);
    }

    public void Forward(double[] input)
    {
        double[] activations = activationFunction.Forward(input);
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
