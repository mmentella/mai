namespace mai.v1;

public interface ILayer
{
    void Backward(double[] input, double[] outputError, double learningRate);
    void Forward(double[] input);

    void Stack(ILayer layer);
    void SetNextLayer(ILayer layer);
    void SetPreviousLayer(ILayer layer);

    double[] Output { get; }
    ILayer? PreviousLayer { get; }
    ILayer? NextLayer { get; }
}