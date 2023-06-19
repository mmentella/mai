namespace mai.v1.functions;

public class MeanSquaredErrorLossFunction
    : LossFunction
{
    public override double[] GradientLoss(double[] actualOutput, double[] expectedOutput)
    {
        double[] output = new double[actualOutput.Length];
        Parallel.For(0, actualOutput.Length, i =>
        {
            output[i] = 0.5 * Math.Pow(expectedOutput[i] - actualOutput[i], 2);
        });
        return output;
    }

    public override double[] Loss(double[] actualOutput, double[] expectedOutput)
    {
        double[] output = new double[actualOutput.Length];
        Parallel.For(0, actualOutput.Length, i =>
        {
            output[i] = actualOutput[i] - expectedOutput[i];
        });
        return output;
    }
}
