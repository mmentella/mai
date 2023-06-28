using mai.v1.activation;
using mai.v1.blas;
using mai.v1.loss;

namespace mai.v1.layers;

public class EmbeddingLayer
    : ILayer
{
    private Matrix output;
    private Matrix weightedInput;
    private Matrix logits;
    private Matrix hiddenWeights;

    private readonly ActivationFunction softmax;
    private readonly LossFunction crossentropy;

    public EmbeddingLayer(int inputLength, int embeddedLength)
    {
        InputSize = inputLength;
        OutputSize = inputLength;

        Weights = new Matrix(inputLength, embeddedLength);
        hiddenWeights = new Matrix(embeddedLength, inputLength);

        Random random = new(20230626);
        Weights = Weights.Run(Weights, x => random.NextDouble());
        hiddenWeights = hiddenWeights.Run(hiddenWeights, x => random.NextDouble());

        softmax = new SoftmaxActivationFunction();
        crossentropy = new CrossEntropyLossFunction();

        output = default!;
        weightedInput = default!;
        logits = default!;
    }

    public int InputSize { get; private set; }
    public int OutputSize { get; private set; }
    public Matrix Weights { get; private set; }

    public Matrix Output => output;
    public Matrix EmbeddedOutput => weightedInput;

    public ILayer? PreviousLayer { get; private set; }
    public ILayer? NextLayer { get; private set; }

    public void Backward(Matrix input, Matrix outputError, double learningRate)
    {
        Matrix dedwh = weightedInput.Transpose() * (output - input);
        Matrix dedw = input.Transpose() * ((output - input) * hiddenWeights.Transpose());

        hiddenWeights -= learningRate * dedwh;
        Weights -= learningRate * dedw;
    }

    public void Forward(Matrix input)
    {
        weightedInput = input * Weights;
        logits = weightedInput * hiddenWeights;

        output = softmax.Forward(logits);
    }

    public void SetNextLayer(ILayer layer)
    {
        NextLayer = layer;
    }

    public void SetPreviousLayer(ILayer layer)
    {
        PreviousLayer = layer;
    }

    public void Stack(ILayer layer)
    {
        NextLayer = layer;
        layer.SetPreviousLayer(this);
    }

    public double GetLoss(Matrix target)
    {
        return crossentropy.Loss(output, target);
    }
}
