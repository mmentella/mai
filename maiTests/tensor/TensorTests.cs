using mai.tensor;
using FluentAssertions;
using System.Linq;
using Xunit;
using System.Collections.Generic;
using System;
using System.Diagnostics;

namespace mai.tensor.Tests
{
    public class TensorTests
    {
        [Fact]
        public void ConstructorTest()
        {
            Tensor tensor = new Tensor(new int[] { 1, 2, 3 });

            Assert.Equal(6, tensor.Length);
            Assert.Equal(6, tensor.Stride[0]);
            Assert.Equal(3, tensor.Stride[1]);
            Assert.Equal(1, tensor.Stride[2]);
        }

        [Fact()]
        public void TensorPrint()
        {
            double[] data = new double[]
            {
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            };

            Tensor tensor = data.AsTensor(new int[] { 3, 3 });

            Assert.Equal(9, tensor.Length);
            Assert.Equal("(3,3)", tensor.Shape.Print());
            Assert.Equal("(3,1)", tensor.Stride.Print());
        }

        [Fact()]
        public void TensorAccessor()
        {
            double[] data = new double[]
            {
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            };

            Tensor tensor = data.AsTensor(new int[] { 3, 3 });

            Assert.Equal(1, tensor[new int[] { 0, 0 }]);
            Assert.Equal(2, tensor[new int[] { 0, 1 }]);
            Assert.Equal(3, tensor[new int[] { 0, 2 }]);
            Assert.Equal(4, tensor[new int[] { 1, 0 }]);
            Assert.Equal(5, tensor[new int[] { 1, 1 }]);
            Assert.Equal(6, tensor[new int[] { 1, 2 }]);
            Assert.Equal(7, tensor[new int[] { 2, 0 }]);
            Assert.Equal(8, tensor[new int[] { 2, 1 }]);
            Assert.Equal(9, tensor[new int[] { 2, 2 }]);
        }

        [Fact()]
        public void FillTest()
        {
            double[] data = new double[]
            {
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            };

            Tensor tensor = data.AsTensor(new int[] { 3, 3 });
            tensor.Fill(1);

            Assert.Equal(1, tensor[0, 0]);
            Assert.Equal(1, tensor[0, 1]);
            Assert.Equal(1, tensor[0, 2]);
            Assert.Equal(1, tensor[1, 0]);
            Assert.Equal(1, tensor[1, 1]);
            Assert.Equal(1, tensor[1, 2]);
            Assert.Equal(1, tensor[2, 0]);
            Assert.Equal(1, tensor[2, 1]);
            Assert.Equal(1, tensor[2, 2]);
        }

        [Fact()]
        public void TransposeTest()
        {
            double[] data = Enumerable.Range(0, 24)
                                      .Select(d => (double)d)
                                      .ToArray();

            Tensor tensor = data.AsTensor(2, 3, 4);
            Tensor reshape = tensor.Reshape(4,2,3);
            Tensor transpose = tensor.Transpose(2,0,1);

            Debug.WriteLine(tensor.Print());
            Debug.WriteLine(reshape.Print());
            Debug.WriteLine(transpose.Print());
        }

        [Fact()]
        public void ReduceTest()
        {
            Tensor tensor =
            Enumerable.Range(1, 24)
                      .Select(r => (double)r)
                      .ToArray()
                      .AsTensor(2, 3, 4);
            Tensor subtensor = tensor.Reduce();
            subtensor.Rank.Should().Be(tensor.Rank - 1);

            subtensor[0, 0].Should().Be(tensor[0, 0, 0]);
            subtensor[0, 1].Should().Be(tensor[0, 0, 1]);
            subtensor[0, 2].Should().Be(tensor[0, 0, 2]);
            subtensor[0, 3].Should().Be(tensor[0, 0, 3]);
        }

        [Fact()]
        public void ReshapeTest()
        {
            Tensor tensor =
            Enumerable.Range(1, 24)
                      .Select(r => (double)r)
                      .ToArray()
                      .AsTensor(2, 3, 4);
            Tensor reshape = tensor.Reshape(4, 3, 2);
            Tensor transpose = tensor.Transpose();

            Debug.WriteLine(tensor.Print());
            Debug.WriteLine(reshape.Print());
        }

        [Fact()]
        public void SliceTest()
        {
            Tensor tensor =
            Enumerable.Range(1, 24)
                      .Select(r => (double)r)
                      .ToArray()
                      .AsTensor(2, 3, 4);
            Tensor slice = tensor.Slice(begin: new int[] { 1, 1, 0 },
                                        size: new int[] { 1, 2, 4 });
            slice[0, 0, 0].Should().Be(tensor[1, 1, 0]);
            slice[0, 1, 3].Should().Be(tensor[1, 2, 3]);
        }

        [Fact()]
        public void ContractionTest()
        {
            Tensor left =
            Enumerable.Range(1, 24)
                      .Select(r => (double)r)
                      .ToArray()
                      .AsTensor(2, 3, 4);
            Tensor right =
            Enumerable.Range(1, 24)
                      .Select(r => (double)r)
                      .ToArray()
                      .AsTensor(2, 3, 4);

            Tensor tensor = left.Contraction(right, new int[][] { new int[] { 1 },
                                                                  new int[] { 1 } });
            Debug.WriteLine(left.Print());
            Debug.WriteLine(right.Print());
            Debug.WriteLine(tensor.Print());
        }

        [Fact()]
        public void DotTest()
        {
            Tensor left =
            Enumerable.Range(1, 4)
                      .Select(r => (double)r)
                      .ToArray()
                      .AsTensor(2, 2);
            Tensor right =
            Enumerable.Range(1, 4)
                      .Select(r => (double)r)
                      .ToArray()
                      .AsTensor(2, 2);

            Tensor tensor = left.Dot(right);

            Debug.WriteLine(left.Print());
            Debug.WriteLine(right.Print());
            Debug.WriteLine(tensor.Print());
        }

        [Fact()]
        public void ToStringTest()
        {
            Tensor tensor =
            Enumerable.Range(100, 100)
                      .Select(r => (double)r)
                      .ToArray()
                      .AsTensor(2, 5, 2, 5);
            string value = tensor.Print();
            Debug.WriteLine(value);
        }
    }
}