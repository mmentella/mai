using mai.v1.tensor;

namespace mai.v1.loss;

public class MeanSquaredErrorLossFunction
    : LossFunction
{
    public override Tensor GradientLoss(Tensor actualOutput, Tensor expectedOutput)
    {
        Tensor output = new(actualOutput.Shape);
        Parallel.For(0, actualOutput.Length, i =>
        {
            output[i] = actualOutput[i] - expectedOutput[i];
        });
        return output;
    }

    public override double Loss(Tensor actualOutput, Tensor expectedOutput)
    {
        double loss = 0;
        for (int i = 0; i < actualOutput.Length; i++)
        {
            loss += Math.Pow(expectedOutput[i] - actualOutput[i], 2);
        }
        return 0.5 * loss;
    }
}