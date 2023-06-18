namespace mai.differentiation;

public interface INode
{
    void Forward(double input);
    void Backward(double outputGradient, double learningRate);

    INode? Next { get; }
    INode? Previous { get; }

    double Input { get; }
    double Output { get; }

    void Connect(INode? next, INode? previous);
}

public class LinearNode : INode
{
    private double w;
    private double d;

    private double output;
    private double input;
    private INode? next;
    private INode? previous;

    public void Forward(double input)
    {
        this.input = input;
        output = w * input + d;

        next?.Forward(output);
    }
    public void Backward(double outputGradient, double learningRate)
    {
        w -= learningRate * input * outputGradient;
        d -= learningRate * outputGradient;

        previous?.Backward(input * outputGradient, learningRate);
    }

    public double Input => input;
    public double Output => output;

    public INode? Next { get => next; protected set => next = value; }
    public INode? Previous { get => previous; protected set => previous = value; }

    public override string ToString()
    {
        return $"{{w: {w:0.0000},d: {d:0.0000}}}";
    }

    public void Connect(INode? next, INode? previous)
    {
        this.next = next;
        this.previous = previous;
    }
}

public class SigmoidNode : INode
{
    private double output;
    private double input;
    private INode? next;
    private INode? previous;

    public void Forward(double input)
    {
        this.input = input;
        output = 1 / (1 + Math.Exp(-input));
    }
    public void Backward(double outputGradient, double learningRate)
    {
        double sigmoid = output;
        double sigmoidGradient = sigmoid * (1 - sigmoid);

        previous?.Backward(outputGradient * sigmoidGradient, learningRate);
    }

    public double Input => input;
    public double Output => output;

    public INode? Next { get => next; protected set => next = value; }

    public INode? Previous { get => previous; protected set => previous = value; }

    public override string ToString()
    {
        return $"{{sigmoid: {output:0.0000}}}";
    }

    public void Connect(INode? next, INode? previous)
    {
        this.next = next;
        this.previous = previous;
    }
}

public class SimpleNetwork
{
    private INode root;
    private INode leaf;

    public SimpleNetwork()
    {
        root = new LinearNode();
        leaf = new SigmoidNode();

        root.Connect(leaf, null);
        leaf.Connect(null, root);
    }

    public void Forward(double input)
    {
        root.Forward(input);
    }

    public void Backward(double outputGradient, double learningRate)
    {
        leaf.Backward(outputGradient, learningRate);
    }

    public double Input => root.Input;

    public double Output => leaf.Output;
}