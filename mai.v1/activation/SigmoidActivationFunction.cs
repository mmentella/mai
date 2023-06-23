namespace mai.v1.activation;

public class SigmoidActivationFunction
    : ActivationFunction
{
    public override double[] Forward(double[] input)
    {
        double[] output = new double[input.Length];
        Parallel.For(0, input.Length, i =>
        {
            output[i] = 1.0 / (1.0 + Math.Exp(-input[i]));
        });
        return output;
    }

    public override double[] Backward(double[] input)
    {
        double[] output = new double[input.Length];
        Parallel.For(0, input.Length, i =>
        {
            double value = 1.0 / (1.0 + Math.Exp(-input[i]));
            output[i] = value * (1 - value);
        });
        return output;
    }
}