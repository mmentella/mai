using mai.blas;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mai.network
{
    public class Trainer
    {
        private readonly Optimizer optimizer;
        private double bestLoss = 1.0e-9;

        public Trainer(Optimizer optimizer)
        {
            this.optimizer = optimizer;
        }

        public IEnumerable<(Matrix samples, Matrix labels)> GenerateBatches(Matrix input, Matrix output, int size = 32)
        {
            if (input.Rows != output.Rows) { throw new ArgumentException(); }

            for (int i = 0; i < input.Rows; i += size)
            {
                yield return (input.GetRows(i, i + size), output.GetRows(i, i + size));
            }
        }

        public void Fit(Matrix samples,
                        Matrix labels,
                        Matrix samplesTest,
                        Matrix labelsTest,
                        int epochs = 100,
                        int checkStep = 10,
                        int batchSize = 32,
                        int seed = 1,
                        bool restart = true)
        {
            if (restart)
            {
                foreach (var layer in optimizer.Network.Layers)
                {
                    layer.First = true;
                }

                bestLoss = 1.0e-9;
            }

            for (int e = 0; e < epochs; e++)
            {
                Random random = new(seed);
                int[] permutation = Enumerable.Range(0, samples.Rows)
                                              .OrderBy(k => random.Next())
                                              .ToArray();
                samples.PermuteRows(permutation);
                labels.PermuteRows(permutation);

                var batches = GenerateBatches(samples, labels, batchSize);
                foreach (var (sample, label) in batches)
                {
                    optimizer.Network.Train(sample, label);
                    optimizer.Step();
                }

                if ((e + 1) % checkStep == 0)
                {
                    var predictions = optimizer.Network.Forward(samplesTest);
                    var loss = optimizer.Network.Loss.Forward(predictions, labelsTest);

                    Console.WriteLine($"Validation loss after {e + 1} epochs is {loss}");
                }
            }
        }
    }
}
