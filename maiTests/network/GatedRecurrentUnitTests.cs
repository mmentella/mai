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
        public void ForwardTest()
        {
            double[] input = Enumerable.Range(1, 100).Select(r => (double)r).ToArray();
            GatedRecurrentUnit gru = new GatedRecurrentUnit(100, 100, 110);

            double[] output = gru.Forward(input);
        }

        [Fact()]
        public void TransposeTest()
        {
            double[] input = Enumerable.Range(1, 12).Select(r => (double)r).ToArray();
            GatedRecurrentUnit gru = new(12, 12, 12);

            var transpose = gru.Transpose(input, 3, 4);

            Debug.WriteLine(input.Print(3, 4));
            Debug.WriteLine(transpose.ToArray().Print(4, 3));
        }

        [Fact()]
        public void TrainTest()
        {
            double[] sin = Enumerable.Range(1, 1000)
                                       .Select(r => Math.Sin(0.1d * r * Math.PI))
                                       .ToArray();
            List<(double[] sample, double[] label)> trainingSet = new();
            for (int s = 5; s <= sin.Length; s++)
            {
                trainingSet.Add((sin[(s - 5)..s], sin[(s - 1)..s]));
            }
            GatedRecurrentUnit gru = new(1, 1, 5);

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

        [Fact()]
        public void HadamardTest()
        {
            double[] values = Enumerable.Range(1, 10)
                                        .Select(i => (double)i)
                                        .ToArray();
            GatedRecurrentUnit gru = new(10, 10, 10);

            var hadamard = gru.Hadamard(values, values, values);
        }

        [Fact()]
        public void DotProductTest()
        {
            double[] values = Enumerable.Range(1, 3)
                                        .Select(i => (double)i)
                                        .ToArray();
            GatedRecurrentUnit gru = new(3, 3, 3);

            var dot = gru.DotProduct(values, values, 3, 1, 1, 3);
        }
    }
}