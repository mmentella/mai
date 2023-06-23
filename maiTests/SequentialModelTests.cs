using mai.differentiation;
using mai.v1.activation;
using mai.v1.layers;
using mai.v1.loss;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Xunit;

namespace mai.v1.Tests
{
    public class SequentialModelTests
    {
        [Fact()]
        public void ForwardTest()
        {
            SequentialModel model = new();
            model.Add(new LinearLayer(15, 5));
            model.Add(new ActivationLayer(new ESwishActivationFunction(1.25)));
            model.Add(new LinearLayer(5, 15));
            model.Add(new ActivationLayer(new SoftmaxActivationFunction()));

            LossFunction lossFunction = new CrossEntropyLossFunction();

            double crossEntropyLoss = 0;
            double[] loss = Array.Empty<double>();
            double[] output;
            List<double[]> input = new(){
                new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
                new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
                new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
                new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
                new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
                new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
                new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
                new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
                new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
            };
            for (int i = 0; i < 10000; i++)
            {
                foreach(var item in input)
                {
                    output = model.Forward(item);

                    loss = lossFunction.Loss(output, item);
                    double[] gradientLoss = lossFunction.GradientLoss(output, item);
                    model.Backward(item, gradientLoss, 0.00001);

                    double prevCrossEntropyLoss = crossEntropyLoss;
                    crossEntropyLoss = loss.Sum();
                    Debug.WriteLine($"epoch: {i:0}|loss: {crossEntropyLoss:0.0000}|prevloss: {prevCrossEntropyLoss:0.0000}");
                }
            }
        }
    }
}