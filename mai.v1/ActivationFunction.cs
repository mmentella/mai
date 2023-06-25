using mai.v1.blas;

namespace mai.v1;

public abstract class ActivationFunction
{
    public abstract Matrix Forward(Matrix input);

    public abstract Matrix Backward(Matrix gradient);
}
