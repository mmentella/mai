using mai.v1.tensor;

namespace mai.v1;

public abstract class LossFunction
{
    public abstract double Loss(Tensor output, Tensor expectedOutput);
    public abstract Tensor GradientLoss(Tensor output, Tensor expectedOutput);
}
