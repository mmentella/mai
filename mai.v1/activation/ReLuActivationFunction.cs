using mai.v1.blas;

namespace mai.v1.activation;

public class ReLuActivationFunction
    : ActivationFunction
{
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
            output[i] = input[i] > 0 ? 1 : 0;
        });
        return output;
    }
}