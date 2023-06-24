using mai.v1.tensor;

namespace mai.v1.loss;

public class CrossEntropyLossFunction
    : LossFunction
{
    public override Tensor GradientLoss(Tensor actualOutput, Tensor expectedOutput)
    {
        Tensor output = new Tensor(actualOutput.Shape);
        Parallel.For(0, actualOutput.Length, i =>
        {
            output[i] = expectedOutput[i] / actualOutput[i];
        });
        return output;
    }

    public override double Loss(Tensor actualOutput, Tensor expectedOutput)
    {
        double loss = 0;
        for (int i = 0; i < actualOutput.Length; i++)
        {
            loss += expectedOutput[i] * Math.Log(actualOutput[i]);
        }
        return -loss;
    }
}
