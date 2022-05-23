using mai.blas;
using mai.network;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Xunit;

namespace mai.network.Tests
{
    public class GatedRecurrentUnitTests
    {
        [Fact()]
        public void TrainTest()
        {
            double[] sin = Enumerable.Range(1, 1000)
                                       .Select(r => Math.Sin(0.1d * r * Math.PI))
                                       .ToArray();
            List<(Matrix sample, Matrix label)> trainingSet = new();
            for (int s = 5; s <= sin.Length; s++)
            {
                trainingSet.Add((new Matrix(sin[(s - 5)..s]), new Matrix(sin[(s - 1)..s])));
            }
            GatedRecurrentUnit gru = new(1, 5);

            double[] loss = gru.Train(trainingSet, k1: 20, k2: 20, epochs: 1);
        }

        private double[] GaussianDistribution(double mean, double stdDev, int length)
        {
            Random rand = new Random();

            double[] distribution = (double[])Array.CreateInstance(typeof(double), length);

            double u1, u2, randStdNormal, randNormal;
            for (int i = 0; i < length; i++)
            {
                u1 = 1.0 - rand.NextDouble();
                u2 = 1.0 - rand.NextDouble();

                randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                       Math.Sin(2.0 * Math.PI * u2);
                randNormal = mean + stdDev * randStdNormal;

                distribution[i] = randNormal;
            }

            return distribution;
        }
    }
}