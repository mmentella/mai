using FluentAssertions;
using mai.blas;
using mai.mnist;
using mai.network;
using System;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.Intrinsics;
using Xunit;

namespace maiTests.network
{
    public class MNISTTest
    {
        [Fact]
        public void MNISTSoftmaxCrossEntropyLossTest()
        {
            
            NeuralNetwork network = new(new Layer[] {new Dense(89, activation:new Tanh()),
                                                     new Dense(10, activation:new Linear())
                                                    }, new SoftmaxCrossEntropyLoss(), seed: 20220603);
            Optimizer sgd = new SGD(network, learningRate: 0.1f);
            Trainer trainer = new(optimizer: sgd);

            var (samples, labels, testSamples, testLabels) = DataProvider.BuildMNIST(60000);
            samples = samples.StandardScale();

            trainer.Fit(samples, labels, testSamples, testLabels, batchSize: 10);
        }
    }
}
