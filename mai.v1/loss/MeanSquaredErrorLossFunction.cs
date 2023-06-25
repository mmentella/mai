using mai.v1.blas;

namespace mai.v1.loss;

public class MeanSquaredErrorLossFunction
    : LossFunction
{
    public override Matrix GradientLoss(Matrix actualOutput, Matrix expectedOutput)
    {
        Matrix output = new(actualOutput.Rows, actualOutput.Columns);
        Parallel.For(0, actualOutput.Length, i =>
        {
            output[i] = actualOutput[i] - expectedOutput[i];
        });
        return output;
    }

    public override double Loss(Matrix actualOutput, Matrix expectedOutput)
    {
        double loss = 0;
        for (int i = 0; i < actualOutput.Length; i++)
        {
            loss += Math.Pow(expectedOutput[i] - actualOutput[i], 2);
        }
        return 0.5 * loss;
    }
}