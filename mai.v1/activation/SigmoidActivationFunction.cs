using mai.v1.tensor;

namespace mai.v1.activation;

public class SigmoidActivationFunction
    : ActivationFunction
{
    public override Tensor Forward(Tensor input)
    {
        Tensor output = new(input.Shape);
        Parallel.For(0, input.Length, i =>
        {
            output[i] = 1.0 / (1.0 + Math.Exp(-input[i]));
        });
        return output;
    }

    public override Tensor Backward(Tensor input)
    {
        Tensor output = new(input.Shape);
        Parallel.For(0, input.Length, i =>
        {
            double value = 1.0 / (1.0 + Math.Exp(-input[i]));
            output[i] = value * (1 - value);
        });
        return output;
    }
}