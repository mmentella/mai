using mai.v1.tensor;

namespace mai.v1;

public interface ILayer
{
    void Backward(Tensor input, Tensor outputError, double learningRate);
    void Forward(Tensor input);

    void Stack(ILayer layer);
    void SetNextLayer(ILayer layer);
    void SetPreviousLayer(ILayer layer);

    Tensor Output { get; }
    ILayer? PreviousLayer { get; }
    ILayer? NextLayer { get; }
}