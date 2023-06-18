using FluentAssertions;
using mai.differentiation;
using Xunit;

namespace maiTests.differentiation
{
    public class OptimizerTest
    {
        [Fact]
        public void TrainTest()
        {
            Optimizer optimizer = new();
            Neuron neuron = optimizer.Train(3, 9, 10000, 0.001);
            
            neuron.Forward(3);

            neuron.Output.Should().BeApproximately(9, 0.02);
        }
    }
}
