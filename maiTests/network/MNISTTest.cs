using mai.mnist;
using mai.network;
using System;
using Xunit;

namespace maiTests.network
{
    public class MNISTTest
    {
        [Fact]
        public void MNISTSoftmaxCrossEntropyLossTest()
        {
            int trainingSamples = 6000;
            int validationSamples = trainingSamples / 6;

            NeuralNetwork network = new(new Layer[] {new Dense(89, activation:new Tanh()),
                                                     new Dense(10, activation:new Linear())
                                                    }, new SoftmaxCrossEntropyLoss(), seed: 20220603);
            Optimizer sgd = new SGD(network, learningRate: 0.1d);
            Trainer trainer = new(optimizer: sgd);

            var (samples, labels, testSamples, testLabels) = DataProvider.BuildMNIST(trainingSamples, validationSamples);
            samples = samples.StandardScale();
            testSamples = testSamples.StandardScale();

            //GC.Collect();

            trainer.Fit(samples, labels, testSamples, testLabels, batchSize: 10, checkStep: 5);
        }
    }
}
