using mai.v1.blas;
using mai.v1;
using mai.v1.activation;
using mai.v1.layers;
using mai.v1.loss;
using System;
using System.Collections.Generic;
using Xunit;
using System.Diagnostics;

namespace maiTests.v1;

public class SequentialModelTests
{
    [Fact()]
    public void ForwardTest()
    {
        SequentialModel model = new();
        model.Add(new CrossEntropySoftmaxLayer(5, 15));

        double crossEntropyLoss;
        bool @break = false;
        Matrix output;
        List<Matrix> input = new(){
            new double[] { 1, 0, 0, 0, 0/*, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0*/ },
            new double[] { 0, 1, 0, 0, 0/*, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0*/ },
            new double[] { 0, 0, 1, 0, 0/*, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0*/ },
            new double[] { 0, 0, 0, 1, 0/*, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0*/ },
            new double[] { 0, 0, 0, 0, 1/*, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0*/ },
            //new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            //new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
            //new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
            //new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
            //new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
            //new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
            //new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
            //new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
            //new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
            //new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
        };
        for (int i = 0; i < 10000; i++)
        {
            crossEntropyLoss = 0;
            input.Shuffle();
            foreach (var item in input)
            {
                output = model.Forward(item);
                model.Backward(item,null)
            }
            Debug.WriteLine($"epoch: {i:0}|loss: {crossEntropyLoss:0.0000}");
            if (@break)
            {
                break;
            }
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