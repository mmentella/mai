using FluentAssertions;
using mai.differentiation;
using Xunit;

namespace maiTests.differentiation;

public class NeuronTest
{
    [Fact]
    public void ForwardTest()
    {
        var neuron = new Neuron();
        var input = 1.0;
        var expected = 0.0;

        neuron.Forward(input);
        neuron.Output.Should().Be(expected);
    }
}
