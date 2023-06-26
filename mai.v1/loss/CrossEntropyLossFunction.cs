using mai.v1.blas;

namespace mai.v1.loss;

public class CrossEntropyLossFunction
    : LossFunction
{
    public override Matrix GradientLoss(Matrix actualOutput, Matrix expectedOutput)
    {
        Matrix output = new(actualOutput.Rows, actualOutput.Columns);
        Parallel.For(0, actualOutput.Length, i =>
        {
            output[i] = expectedOutput[i] / actualOutput[i];
        });
        return output;
    }

    public override double Loss(Matrix actualOutput, Matrix expectedOutput)
    {
        double loss = 0;
        for (int i = 0; i < actualOutput.Length; i++)
        {
            loss += -expectedOutput[i] * Math.Log(actualOutput[i]);
        }
        return loss;
    }
}
