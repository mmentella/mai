namespace mai.v1.activation;

public class ReLuActivationFunction
    : ActivationFunction
{
    public override double[] Forward(double[] input)
    {
        double[] output = new double[input.Length];
        Parallel.For(0, input.Length, i =>
        {
            output[i] = Math.Max(0, input[i]);
        });
        return output;
    }

    public override double[] Backward(double[] input)
    {
        double[] output = new double[input.Length];
        Parallel.For(0, input.Length, i =>
        {
            output[i] = input[i] > 0 ? 1 : 0;
        });
        return output;
    }
}