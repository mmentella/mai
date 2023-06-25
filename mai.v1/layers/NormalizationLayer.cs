using mai.v1.blas;

namespace mai.v1.layers;

public class NormalizationLayer
    : ILayer
{
    public NormalizationLayer()
    {
        Output = default!;
    }

    public Matrix Output { get; private set; }

    public ILayer? PreviousLayer { get; private set; }
    public ILayer? NextLayer { get; private set; }

    public void Backward(Matrix input, Matrix outputError, double learningRate)
    {
        PreviousLayer?.Backward(input, outputError, learningRate);
    }

    public void Forward(Matrix input)
    {
        Matrix normalizedInput = Normalize(input);
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

    private static Matrix Normalize(Matrix input)
    {
        double mean = Mean(input);
        double std = Std(input, mean);
        Matrix output = new(input.Rows, input.Columns);
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = (input[i] - mean) / std;
        }
        return output;
    }

    private static double Std(Matrix input, double mean)
    {
        double sum = 0;
        for (int i = 0; i < input.Length; i++)
        {
            sum += Math.Pow(input[i] - mean, 2);
        }
        return Math.Sqrt(sum / input.Length);
    }

    private static double Mean(Matrix input)
    {
        double sum = 0;
        for (int i = 0; i < input.Length; i++)
        {
            sum += input[i];
        }
        return sum / input.Length;
    }
}
