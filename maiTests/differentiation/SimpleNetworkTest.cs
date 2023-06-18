using FluentAssertions;
using mai.differentiation;
using Microsoft.VisualStudio.TestPlatform.Utilities;
using System;
using System.Diagnostics;
using Xunit;

namespace maiTests.differentiation;

public class SimpleNetworkTest
{
    [Fact]
    public void ForwardTest()
    {
        SimpleNetwork network = new();
        network.Forward(1);

        network.Input.Should().Be(1);
        network.Output.Should().Be(0.5);
    }

    [Fact]
    public void TrainTest()
    {
        SimpleNetwork network = new();

        for (int i = 0; i < 1000000; i++)
        {
            network.Forward(1);
            
            double loss = 0.5 * Math.Pow(1 - network.Output, 2);
            if (loss < 0.0001)
            {
                break;
            }

            double outputGradient = network.Output - 1;
            network.Backward(outputGradient, 0.001);

            Debug.WriteLine($"epoch: {i:0}|output: {1:0}|prediction: {network.Output:0.0000}|loss: {loss:0.0000}");
        }
    }
}
