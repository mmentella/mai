using mai.v1;
using mai.v1.functions;
using mai.v1.layers;
using System;
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
        ILayer activationLayer = new ActivationLayer(new ESwishActivationFunction(1.5));
        LossFunction lossFunction = new MeanSquaredErrorLossFunction();

        linearLayer.Stack(activationLayer);

        double[] input = new double[1];
        double[] output = new double[1];

        // Act
        for (int epoch = 0; epoch < 1000; epoch++)
        {
            for (double i = 0; i <= 1; i += 0.1)
            {
                input[0] = i;
                output[0] = Math.Exp(-0.5 * i * i) / (Math.Sqrt(2 * Math.PI));

                linearLayer.Forward(input);
                double[] prediction = activationLayer.Output;

                double[] loss = lossFunction.Loss(prediction, output);
                linearLayer.Backward(input, lossFunction.GradientLoss(prediction, output), 0.001);

                Debug.WriteLine($"epoch: {epoch:0}|output: {output[0]:0.0000}|prediction: {prediction[0]:0.0000}|loss: {loss[0]:0.0000}");
            }
        }
    }
}
