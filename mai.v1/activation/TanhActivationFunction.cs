using mai.v1.blas;

namespace mai.v1.activation;

public class TanhActivationFunction
    : ActivationFunction
{
    public override Matrix Forward(Matrix input)
    {
        Matrix output = new(input.Rows, input.Columns);
        Parallel.For(0, input.Length, i =>
        {
            output[i] = Math.Tanh(input[i]);
        });
        return output;
    }

    public override Matrix Backward(Matrix input)
    {
        Matrix output = new(input.Rows, input.Columns);
        Parallel.For(0, input.Length, i =>
        {
            double value = Math.Tanh(input[i]);
            output[i] = 1 - value * value;
        });
        return output;
    }
}