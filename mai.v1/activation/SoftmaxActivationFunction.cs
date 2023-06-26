using mai.v1.blas;

namespace mai.v1.activation;

public class SoftmaxActivationFunction
    : ActivationFunction
{
    private Matrix output = default!;

    public override Matrix Forward(Matrix input)
    {
        output = new Matrix(input.Rows, input.Columns);
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

    public override Matrix Backward(Matrix gradient)
    {
        Matrix outputGradient = new(output.Columns, output.Columns);
        for (int i = 0; i < outputGradient.Rows; i++)
        {
            for (int j = 0; j < outputGradient.Columns; j++)
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