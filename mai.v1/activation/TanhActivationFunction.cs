using mai.v1.tensor;

namespace mai.v1.activation;

public class TanhActivationFunction
    : ActivationFunction
{
    public override Tensor Forward(Tensor input)
    {
        Tensor output = new(input.Shape);
        Parallel.For(0, input.Length, i =>
        {
            output[i] = Math.Tanh(input[i]);
        });
        return output;
    }

    public override Tensor Backward(Tensor input)
    {
        Tensor output = new(input.Shape);
        Parallel.For(0, input.Length, i =>
        {
            double value = Math.Tanh(input[i]);
            output[i] = 1 - value * value;
        });
        return output;
    }
}