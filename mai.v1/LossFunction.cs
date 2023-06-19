namespace mai.v1;

public abstract class LossFunction
{
    public abstract double[] Loss(double[] output, double[] expectedOutput);
    public abstract double[] GradientLoss(double[] output, double[] expectedOutput);
}
