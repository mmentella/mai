using mai.v1;
using mai.v1.blas;
using mai.v1.layers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using Xunit;

namespace maiTests.v1;

public class SequentialModelTests
{
    [Fact()]
    public void ForwardTest()
    {
        EmbeddingLayer embeddingLayer = new(15, 5);

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
}