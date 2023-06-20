using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mai.v1.layers;

public class NormalizationLayer
    : ILayer
{
    public NormalizationLayer()
    {
        Output = Array.Empty<double>();
    }

    public double[] Output { get; private set; }

    public ILayer? PreviousLayer { get; private set; }
    public ILayer? NextLayer { get; private set; }

    public void Backward(double[] input, double[] outputError, double learningRate)
    {
        PreviousLayer?.Backward(input, outputError, learningRate);
    }

    public void Forward(double[] input)
    {
        double[] normalizedInput = Normalize(input);
        Output = normalizedInput;
        NextLayer?.Forward(Output);
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

    private static double[] Normalize(double[] input)
    {
        double mean = Mean(input);
        double std = Std(input, mean);
        double[] output = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = (input[i] - mean) / std;
        }
        return output;
    }

    private static double Std(double[] input, double mean)
    {
        double sum = 0;
        for (int i = 0; i < input.Length; i++)
        {
            sum += Math.Pow(input[i] - mean, 2);
        }
        return Math.Sqrt(sum / input.Length);
    }

    private static double Mean(double[] input)
    {
        double sum = 0;
        for (int i = 0; i < input.Length; i++)
        {
            sum += input[i];
        }
        return sum / input.Length;
    }
}
