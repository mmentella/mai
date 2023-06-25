using mai.v1;
using mai.v1.activation;
using mai.v1.layers;
using mai.v1.loss;
using System;
using System.Collections.Generic;
using Xunit;

namespace maiTests.v1;

public class SequentialModelTests
{
    [Fact()]
    public void ForwardTest()
    {
        SequentialModel model = new();
        model.Add(new LinearLayer(15, 5));
        model.Add(new ActivationLayer(new ESwishActivationFunction(1.25)));
        model.Add(new LinearLayer(5, 15));
        model.Add(new ActivationLayer(new SoftmaxActivationFunction()));

        LossFunction lossFunction = new CrossEntropyLossFunction();

        //double crossEntropyLoss;
        //bool @break = false;
        //Tensor output;
        //List<Tensor> input = new(){
        //    new(new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },new int[]{1,15}),
        //    new(new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },new int[]{1,15}),
        //    new(new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },new int[]{1,15}),
        //    new(new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },new int[]{1,15}),
        //    new(new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },new int[]{1,15}),
        //    new(new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },new int[]{1,15}),
        //    new(new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },new int[]{1,15}),
        //    new(new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },new int[]{1,15}),
        //    new(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },new int[]{1,15}),
        //    new(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 },new int[]{1,15}),
        //    new(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },new int[]{1,15}),
        //    new(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },new int[]{1,15}),
        //    new(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },new int[]{1,15}),
        //    new(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },new int[]{1,15}),
        //    new(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },new int[]{1,15}),
        //};
        //for (int i = 0; i < 10000; i++)
        //{
        //    crossEntropyLoss = 0;
        //    input.Shuffle();
        //    foreach (var item in input)
        //    {
        //        output = model.Forward(item);

        //        double loss = lossFunction.Loss(output, item);
        //        if (double.IsNaN(loss))
        //        {
        //            @break = true;
        //            break;
        //        }
        //        Tensor gradientLoss = lossFunction.GradientLoss(output, item);
        //        model.Backward(item, gradientLoss, 0.001);

        //        crossEntropyLoss += loss;
        //    }
        //    Debug.WriteLine($"epoch: {i:0}|loss: {crossEntropyLoss:0.0000}");
        //    if (@break)
        //    {
        //        break;
        //    }
        //}
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