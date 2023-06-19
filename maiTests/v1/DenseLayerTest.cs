using mai.v1;
using mai.v1.functions;
using mai.v1.layers;
using System.Diagnostics;
using Xunit;

namespace maiTests.v1;

public class DenseLayerTest
{
    [Fact]
    public void LinearLayerSigmoidLayerTest()
    {
        // Arrange
        ILayer linearLayer = new LinearLayer(1, 1);
        ILayer activationLayer = new ActivationLayer(new SigmoidActivationFunction());
        LossFunction lossFunction = new MeanSquaredErrorLossFunction();

        linearLayer.Stack(activationLayer);

        double[] input = new double[1];
        double[] output = new double[1];

        // Act
        for (int epoch = 0; epoch < 10000; epoch++)
        {
            for (int i = -10; i <= 10; i++)
            {
                input[0] = i;
                output[0] = i * i;

                linearLayer.Forward(input);
                double[] prediction = activationLayer.Output;

                double[] error = lossFunction.Loss(prediction, output);
                activationLayer.Backward(prediction, error, 0.001);

                Debug.WriteLine($"epoch: {epoch:0}|output: {output[0]:0}|prediction: {prediction[0]:0.0000}|loss: {error[0]:0.0000}");
            }
        }
    }
}
