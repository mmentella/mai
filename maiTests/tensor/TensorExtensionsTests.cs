using FluentAssertions;
using System.Diagnostics;
using Xunit;

namespace mai.tensor.Tests
{
    public class TensorExtensionsTests
    {
        [Fact]
        public void PrintTest()
        {
            Tensor tensor = new Tensor(new int[] { 1, 2, 3 });
            Debug.WriteLine(tensor.Shape.Print());
            Debug.WriteLine(tensor.Stride.Print());
        }

        [Fact]
        public void LessThanOrEqualsTest()
        {
            int[] left = { 1, 2, 2 };
            int[] right = { 1, 2, 3 };

            Assert.True(left.LessThanOrEquals(right));
        }

        [Fact]
        public void LessThanTest()
        {
            int[] left = { 1, 2, 3 };
            int[] right = { 4, 5, 6 };

            Assert.True(left.LessThanOrEquals(right));
        }

        [Fact]
        public void GreatherThanOrEqualsTest()
        {
            int[] left = { 1, 2, 3 };
            int[] right = { 4, 5, 6 };

            Assert.True(right.GreatherThanOrEquals(left));
        }

        [Fact]
        public void GreatherThanTest()
        {
            int[] left = { 1, 2, 3 };
            int[] right = { 4, 5, 6 };

            Assert.True(right.GreatherThan(left));
        }

        [Fact]
        public void AreEqualsTest()
        {
            int[] left = { 1, 2, 3 };
            int[] right = { 1, 2, 3 };

            Assert.True(left.AreEquals(right));
        }

        [Fact()]
        public void ScalarAsTensorTest()
        {
            Tensor tensor = 1d.AsTensor();

            tensor.Rank.Should().Be(0);
            tensor.Length.Should().Be(0);
        }
    }
}