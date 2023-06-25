using mai.v1.blas;

namespace mai.v1;

public abstract class LossFunction
{
    public abstract double Loss(Matrix output, Matrix expectedOutput);
    public abstract Matrix GradientLoss(Matrix output, Matrix expectedOutput);
}
