﻿using mai.mnist;
using mai.network;
using Xunit;

namespace maiTests.network
{
    public class MNISTTest
    {
        [Fact]
        public void MNISTSoftmaxCrossEntropyLossTest()
        {
            int trainingSamples = 100;
            int validationSamples = trainingSamples / 6;

            NeuralNetwork network = new(new Layer[] {new Dense(89, activation:new Sigmoid(), dropout:1),
                                                     //new Dense(46, activation:new Sigmoid(), dropout:0.9),
                                                     new Dense(10, activation:new Linear())
                                                    }, new SoftmaxCrossEntropyLoss(), seed: 20220603);
            Optimizer sgd = new SGDMomentum(network, learningRate: 0.9d);
            Trainer trainer = new(optimizer: sgd);

            var (samples, labels, testSamples, testLabels) = DataProvider.BuildMNIST(trainingSamples, validationSamples);
            samples = samples.StandardScale();
            testSamples = testSamples.StandardScale();

            //GC.Collect();

            trainer.Fit(samples, labels, testSamples, testLabels, batchSize: 10, checkStep: 5);
        }
    }
}
