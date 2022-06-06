using FluentAssertions;
using mai.blas;
using mai.mnist;
using mai.network;
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
        public void SIMDSumTest()
        {
            Matrix matrix = (Matrix)(Enumerable.Range(1, 10000).Select(r => (double)r).ToArray());

            double sum = matrix.Sum();
            double simdsum = matrix.SIMDSum();

            simdsum.Should().Be(sum);
        }

        [Fact]
        public void ImplicitCastExplicitCastReshapeTest()
        {
            double[] data = { 1d, 2d, 3d, 4d, 5d, 6d };
            Matrix matrix = (Matrix)data;
            matrix.Reshape(2, 3);

            matrix.Rows.Should().Be(2);
            matrix.Columns.Should().Be(3);

            data = (double[])matrix;

            data.Length.Should().Be(6);
        }

        [Fact]
        public void MNISTSoftmaxCrossEntropyLossTest()
        {
            NeuralNetwork network = new(new Layer[] {new Dense(89, activation:new Tanh()),
                                                     new Dense(10, activation:new Linear())
                                                    }, new SoftmaxCrossEntropyLoss(), seed: 20220603);
            Optimizer sgd = new SGD(network, learningRate: 0.1);
            Trainer trainer = new(optimizer: sgd);

            var (samples, labels, testSamples, testLabels) = DataProvider.BuildMNIST();
            samples = samples.StandardScale();

            trainer.Fit(samples, labels, testSamples, testLabels, batchSize: 60);
        }
    }
}
