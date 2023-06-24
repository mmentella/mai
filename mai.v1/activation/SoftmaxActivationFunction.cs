using mai.v1.tensor;

namespace mai.v1.activation;

public class SoftmaxActivationFunction
    : ActivationFunction
{
    private Tensor output = default!;

    public override Tensor Forward(Tensor input)
    {
        output = new Tensor(input.Shape);
        double sum = 0;
        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Math.Exp(input[i]);
            sum += output[i];
        }

        for (int i = 0; i < input.Length; i++)
        {
            output[i] /= sum;
        }

        return output;
    }

    public override Tensor Backward(Tensor gradient)
    {
        Tensor outputGradient = new(output.Shape[0], output.Shape[0]);
        for (int i = 0; i < gradient.Length; i++)
        {
            for (int j = 0; j < gradient.Length; j++)
            {
                if (i == j)
                {
                    outputGradient[i, j] = gradient[i] * output[i] * (1 - output[i]);
                    continue;
                }
                outputGradient[i, j] = -output[i] * output[j] * gradient[i];
            }
        }
        return outputGradient;
    }
}