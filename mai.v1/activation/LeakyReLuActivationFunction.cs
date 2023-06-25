using mai.v1.blas;

namespace mai.v1.activation;

public class LeakyReLuActivationFunction
    : ActivationFunction
{
    public double Slope { get; }

    public LeakyReLuActivationFunction(double slope)
    {
        Slope = slope;
    }

    public override Matrix Forward(Matrix input)
    {
        Matrix output = new(input.Rows, input.Columns);
        Parallel.For(0, input.Length, i =>
        {
            output[i] = Math.Max(0, input[i]);
        });
        return output;
    }

    public override Matrix Backward(Matrix input)
    {
        Matrix output = new(input.Rows, input.Columns);
        Parallel.For(0, input.Length, i =>
        {
            output[i] = input[i] > 0 ? 1 : Slope;
        });
        return output;
    }
}