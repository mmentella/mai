namespace mai.v1.loss;

public class MeanSquaredErrorLossFunction
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

    public override double Loss(double[] actualOutput, double[] expectedOutput)
    {
        double loss = 0;
        for (int i = 0; i < actualOutput.Length; i++)
        {
            loss += Math.Pow(expectedOutput[i] - actualOutput[i], 2);
        }
        return 0.5 * loss;
    }
}