using Xunit;
using mai.network;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mai.blas;

namespace mai.network.Tests
{
    public class TrainerTests
    {
        [Fact()]
        public void FitTest()
        {
            int seed = 20220529;
            Random random = new(seed);
            NeuralNetwork network = new(new Layer[] { new Dense(32, new Tanh(), seed),
                                                      new Dense(32, new Sigmoid(), seed),
                                                      new Dense(32, new Linear(), seed)
                                                    },
                                        new MeanSquaredError(),
                                        seed);
            double[] sin = Enumerable.Range(0, 1024)
                                     .Select(r => Math.Sin(r) + 0.05 * random.NextDouble())
                                     .ToArray();
            Matrix samples = new(sin, 32, 32);
            sin = Enumerable.Range(0, 1024)
                            .Select(r => Math.Sin(r))
                            .ToArray();
            Matrix labels = new(sin, 32, 32);
            sin = Enumerable.Range(0, 128)
                            .Select(r => Math.Sin(r))
                            .ToArray();
            Matrix samplesTest = new(sin, 4, 32);
            Matrix labelsTest = new(sin, 4, 32);

            Optimizer optimizer = new SGD(network, 0.01);
            Trainer trainer = new(optimizer);

            trainer.Fit(samples, labels, samplesTest, labelsTest, batchSize: 1);
        }
    }
}