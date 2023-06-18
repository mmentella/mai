namespace mai.differentiation;

public class Neuron
{
    private double w;
    private double d;

    private double output;
    private double input;

    public void Forward(double input)
    {
        this.input = input;
        output = w * input + d;
    }
    public void Backward(double outputGradient, double learningRate)
    {
        w -= learningRate * input * outputGradient;
        d -= learningRate * outputGradient;
    }

    public double Input => input;
    public double Output => output;

    public override string ToString()
    {
        return $"{{w: {w:0.0000},d: {d:0.0000}}}";
    }
}
