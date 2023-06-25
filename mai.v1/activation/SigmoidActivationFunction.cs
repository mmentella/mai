using mai.v1.blas;

namespace mai.v1.activation;

public class SigmoidActivationFunction
    : ActivationFunction
{
    public override Matrix Forward(Matrix input)
    {
        Matrix output = new(input.Rows, input.Columns);
        Parallel.For(0, input.Length, i =>
        {
            output[i] = 1.0 / (1.0 + Math.Exp(-input[i]));
        });
        return output;
    }

    public override Matrix Backward(Matrix input)
    {
        Matrix output = new(input.Rows, input.Columns);
        Parallel.For(0, input.Length, i =>
        {
            double value = 1.0 / (1.0 + Math.Exp(-input[i]));
            output[i] = value * (1 - value);
        });
        return output;
    }
}