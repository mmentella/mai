using mai.differentiation;
using Xunit;

namespace maiTests.differentiation
{
    public class SigmoidTest
    {
        [Fact]
        public void ForwardTest()
        {
            Assert.Equal(0.5, Sigmoid.Forward(0));
            Assert.Equal(0.7310585786300049, Sigmoid.Forward(1));
            Assert.Equal(0.2689414213699951, Sigmoid.Forward(-1));
        }

        [Fact]
        public void BackwardTest()
        {
            Assert.Equal(0.25, Sigmoid.Backward(0));
            Assert.Equal(0.19661193324148185, Sigmoid.Backward(1));
            Assert.Equal(0.19661193324148185, Sigmoid.Backward(-1));
        }
    }
}
