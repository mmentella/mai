using mai.blas;
using System.Diagnostics;

namespace mai.network
{
    public class Trainer
    {
        private readonly Optimizer optimizer;
        private double bestLoss = 1.0e-9f;

        public Trainer(Optimizer optimizer)
        {
            this.optimizer = optimizer;
        }

        public IEnumerable<(Matrix samples, Matrix labels)> GenerateBatches(Matrix input, Matrix output, int size = 32)
        {
            if (input.Rows != output.Rows) { throw new ArgumentException(); }

            for (int i = 0; i < input.Rows; i += size)
            {
                yield return (input.GetRows(i, size), output.GetRows(i, size));
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

                bestLoss = 1.0e-9f;
            }

            for (int e = 0; e < epochs; e++)
            {
                Random random = new(seed);
                int[] permutation = Enumerable.Range(0, samples.Rows)
                                              .OrderBy(k => random.Next())
                                              .ToArray();
                samples = samples.PermuteRows(permutation);
                labels = labels.PermuteRows(permutation);

                //GC.Collect();

                double loss;
                int batchCount = 0;
                Stopwatch sw = new();
                var batches = GenerateBatches(samples, labels, batchSize);
                foreach (var (sample, label) in batches)
                {
                    sw.Reset();
                    sw.Start();

                    loss = optimizer.Network.Train(sample, label);
                    optimizer.Step();

                    sw.Stop();

                    batchCount++;
                }
                if (e == 0)
                {
                    var predictions = optimizer.Network.Forward(samplesTest);
                    loss = optimizer.Network.Loss.Forward(predictions, labelsTest);

                    Debug.WriteLine($"Initial Validation loss is {loss}");
                }
                if ((e + 1) % checkStep == 0)
                {
                    var predictions = optimizer.Network.Forward(samplesTest);
                    loss = optimizer.Network.Loss.Forward(predictions, labelsTest);

                    Debug.WriteLine($"Validation loss after {e + 1} epochs is {loss}");
                }

                optimizer.Update(e);
            }

            Debug.WriteLine($"Training ended.");
        }
    }
}
