namespace mai.v1.activation;

public class SoftmaxActivationFunction
    : ActivationFunction
{
    private double[] output = Array.Empty<double>();

    public override double[] Forward(double[] input)
    {
        IEnumerable<double> exp = input.Select(Math.Exp);
        double sum = 1d / exp.Sum();

        output = exp.Select(x => x * sum).ToArray();

        return output;
    }

    public override double[] Backward(double[] gradient)
    {
        double[] outputGradient = new double[gradient.Length];
        for (int i = 0; i < gradient.Length; i++)
        {
            outputGradient[i] = gradient[i] * (output[i] * (1 - output[i]));
        }
        return outputGradient;
    }
}