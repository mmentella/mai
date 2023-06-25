using mai.v1.blas;

namespace mai.v1;

public interface ILayer
{
    void Backward(Matrix input, Matrix outputError, double learningRate);
    void Forward(Matrix input);

    void Stack(ILayer layer);
    void SetNextLayer(ILayer layer);
    void SetPreviousLayer(ILayer layer);

    Matrix Output { get; }
    ILayer? PreviousLayer { get; }
    ILayer? NextLayer { get; }
}