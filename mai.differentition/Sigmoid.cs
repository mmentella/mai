namespace mai.differentition;

public class Sigmoid
{
    public double Forward(double x) => 1 / (1 + Math.Exp(-x));

    public double Backward(double x) => Forward(x) * (1 - Forward(x));

    public double[] Backward(double[] x)
    {
        double[] result = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            result[i] = Backward(x[i]);
        }
        return result;
    }

    public double[] Forward(double[] x)
    {
        double[] result = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            result[i] = Forward(x[i]);
        }
        return result;
    }
}