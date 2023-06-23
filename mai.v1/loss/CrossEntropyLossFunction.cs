namespace mai.v1.loss;

public class CrossEntropyLossFunction
    : LossFunction
{
    public override double[] GradientLoss(double[] actualOutput, double[] expectedOutput)
    {
        double[] output = new double[actualOutput.Length];
        Parallel.For(0, actualOutput.Length, i =>
        {
            output[i] = actualOutput[i] - expectedOutput[i];
        });
        return output;
    }

    public override double[] Loss(double[] actualOutput, double[] expectedOutput)
    {
        double[] output = new double[actualOutput.Length];
        Parallel.For(0, actualOutput.Length, i =>
        {
            output[i] = -1 * (expectedOutput[i] * Math.Log(actualOutput[i]));
        });
        return output;
    }
}
