namespace mai.v1;

public abstract class ActivationFunction
{
    public abstract double[] Forward(double[] input);

    public abstract double[] Backward(double[] gradient);
}
