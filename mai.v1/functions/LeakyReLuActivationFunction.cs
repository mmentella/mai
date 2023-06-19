namespace mai.v1.functions;

public class LeakyReLuActivationFunction
    : ActivationFunction
{
    public double Slope { get; }

    public LeakyReLuActivationFunction(double slope)
    {
        Slope = slope;
    }

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
            output[i] = input[i] > 0 ? 1 : Slope;
        });
        return output;
    }
}