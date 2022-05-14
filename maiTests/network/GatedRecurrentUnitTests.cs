using mai.network;
using System;
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
            GatedRecurrentUnit gru = new GatedRecurrentUnit(100, 110);

            double[] output = gru.Forward(input);
        }

        [Fact()]
        public void TransposeTest()
        {
            double[] input = Enumerable.Range(1, 12).Select(r => (double)r).ToArray();
            GatedRecurrentUnit gru = new(12, 12);

            var transpose = gru.Transpose(input, 3, 4);

            Debug.WriteLine(input.Print(3, 4));
            Debug.WriteLine(transpose.ToArray().Print(4, 3));
        }

        [Fact()]
        public void TrainTest()
        {
            double[] input = Enumerable.Range(1, 12).Select(r => (double)r).ToArray();
            double[] output = GaussianDistribution(0, 1, 12);

            GatedRecurrentUnit gru = new(12, 12);

            double[] loss = gru.Train(input, output, learningRate: 0.1d, epochs: 1000);
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