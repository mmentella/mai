using mai.v1.blas;

namespace mai.v1.activation;

public class ESwishActivationFunction
    : ActivationFunction
{
    public double Beta { get; }

    public ESwishActivationFunction(double beta)
    {
        Beta = beta;
    }
    public override Matrix Forward(Matrix input)
    {
        Matrix output = new(input.Rows, input.Columns);
        Parallel.For(0, input.Length, i =>
        {
            output[i] = Beta * input[i] / (1.0 + Math.Exp(-input[i]));
        });
        return output;
    }

    public override Matrix Backward(Matrix input)
    {
        Matrix output = new(input.Rows, input.Columns);
        Parallel.For(0, input.Length, i =>
        {
            double sigmoid = 1.0 / (1.0 + Math.Exp(-input[i]));
            double eswish = Beta * input[i] * sigmoid;
            output[i] = eswish + sigmoid * (Beta - eswish);
        });
        return output;
    }
}