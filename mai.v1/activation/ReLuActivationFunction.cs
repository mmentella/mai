using mai.v1.tensor;

namespace mai.v1.activation;

public class ReLuActivationFunction
    : ActivationFunction
{
    public override Tensor Forward(Tensor input)
    {
        Tensor output = new(input.Shape);
        Parallel.For(0, input.Length, i =>
        {
            output[i] = Math.Max(0, input[i]);
        });
        return output;
    }

    public override Tensor Backward(Tensor input)
    {
        Tensor output = new(input.Shape);
        Parallel.For(0, input.Length, i =>
        {
            output[i] = input[i] > 0 ? 1 : 0;
        });
        return output;
    }
}