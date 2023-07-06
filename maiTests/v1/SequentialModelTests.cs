using FluentAssertions;
using mai.v1.blas;
using mai.v1.layers;
using mai.v1.models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace maiTests.v1;

public class SequentialModelTests
{
    [Fact()]
    public void ForwardTest()
    {
        EmbeddingLayer embeddingLayer = new(15, 2);

        SequentialModel model = new();
        model.Add(embeddingLayer);

        List<Matrix> input = new(){
            new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
        };
        for (int i = 0; i < 10000; i++)
        {
            input.Shuffle();
            double loss = 0;
            foreach (var item in input)
            {
                _ = model.Forward(item);
                model.Backward(item, default!, 0.05);

                loss += embeddingLayer.GetLoss(item);
            }

            Debug.WriteLine($"epoch: {i:0}|loss: {loss:0.0000}");
        }

        input.HotEncodedOrder();

        _ = model.Forward(input[0]);
        Matrix firstEncoded = embeddingLayer.EmbeddedOutput;

        _ = model.Forward(input[1]);
        Matrix secondEncoded = embeddingLayer.EmbeddedOutput;
    }

    [Fact]
    public async Task EurUsdMinutetrade()
    {
        CultureInfo cultureInfo = new("en-US");
        string[] lines = await File.ReadAllLinesAsync(@"C:\Users\Foxyh\source\github\mmentella\mai\maiTests\v1\EUR.USD-Minute-Trade.txt");
        double[] prices =
            lines.Select(l => l.Split(','))
                 .Select(l => new double[] { double.Parse(l[2], cultureInfo), double.Parse(l[5], cultureInfo) })
                 .SelectMany(p => p)
                 .Take(1000)
                 .ToArray();
        IDictionary<double, double[]> lookup = prices.HotEncodeLookup();

        int window = 1;
        IList<(double[] input, double[] output)> trainingSet = new List<(double[] input, double[] output)>();
        for (int d = window; d < prices.Length - window; d++)
        {
            for (int i = -window; i <= window; i++)
            {
                if (i == 0) { continue; }
                trainingSet.Add((lookup[prices[d]], lookup[prices[d + i]]));
            }
        }

        EmbeddingLayer embeddingLayer = new(trainingSet[0].input.Length, 10);

        SequentialModel model = new();
        model.Add(embeddingLayer);

        for (int epoch = 0; epoch < 10000; epoch++)
        {
            double loss = 0;
            foreach (var (input, output) in trainingSet)
            {
                _ = model.Forward(input);
                model.Backward(output, default!, 0.01);

                loss += embeddingLayer.GetLoss(output);
            }

            Debug.WriteLine($"epoch: {epoch:0}|loss: {loss:0.0000}");
        }
    }

    [Fact]
    public void GetPositionEncoding()
    {
        int sequenceLength = 10;
        int embeddingSize = 4;
        int n = 10000;

        Matrix positionEncoding = new(sequenceLength, embeddingSize);
        for (int i = 0; i < sequenceLength; i++)
        {
            for (int j = 0; j < 0.5 * embeddingSize; j++)
            {
                positionEncoding[i, 2 * j] = Math.Sin(i / Math.Pow(n, 2 * i / embeddingSize));
                positionEncoding[i, 2 * j + 1] = Math.Cos(i / Math.Pow(n, 2 * i / embeddingSize));
            }
        }
        Debug.WriteLine(positionEncoding.Print());
    }

    [Fact]
    public void ShuffleAndOrderTest()
    {
        List<Matrix> input = new(){
            new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
            new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
        };

        input.Shuffle();
        input.HotEncodedOrder();

        input[0][0].Should().Be(1);
    }
}

public static class Utility
{
    public static void Shuffle<T>(this IList<T> list)
    {
        Random rng = new();
        int n = list.Count;
        while (n > 1)
        {
            n--;
            int k = rng.Next(n + 1);
            (list[n], list[k]) = (list[k], list[n]);
        }
    }

    public static void HotEncodedOrder(this IList<Matrix> list)
    {
        for (int i = 0; i < list.Count; i++)
        {
            Matrix? hot = list.Where(l => l[i] == 1)
                              .SingleOrDefault();
            if (hot is null) { continue; }

            int indexOf = list.IndexOf(hot);
            (list[i], list[indexOf]) = (list[indexOf], list[i]);
        }
    }

    public static IDictionary<double, double[]> HotEncodeLookup(this double[] prices)
    {
        IDictionary<double, double[]> lookup = new Dictionary<double, double[]>();
        IEnumerable<double> unique = prices.Distinct()
                                           .Order();
        for (int i = 0; i < unique.Count(); i++)
        {
            double[] hot = new double[unique.Count()];
            hot[i] = 1;
            lookup.Add(unique.ElementAt(i), hot);
        }

        return lookup;
    }
}