namespace mai.v1.functions;

public class TanhActivationFunction
    : ActivationFunction
{
    public override double[] Forward(double[] input)
    {
        double[] output = new double[input.Length];
        Parallel.For(0, input.Length, i =>
        {
            output[i] = Math.Tanh(input[i]);
        });
        return output;
    }

    public override double[] Backward(double[] input)
    {
        double[] output = new double[input.Length];
        Parallel.For(0, input.Length, i =>
        {
            double value = Math.Tanh(input[i]);
            output[i] = 1 - value * value;
        });
        return output;
    }
}