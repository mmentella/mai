using mai.v1.tensor;

namespace mai.v1;

public abstract class ActivationFunction
{
    public abstract Tensor Forward(Tensor input);

    public abstract Tensor Backward(Tensor gradient);
}
