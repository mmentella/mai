using FluentAssertions;
using System;
using System.Linq;
using Xunit;

namespace mai.blas.Tests
{
    public class MatrixTests
    {
        [Fact()]
        public void MatrixTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void ReshapeTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void GetRowsTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void TransposeTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void SquareTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void HadamardTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void PermuteRowsTest()
        {
            Matrix test = new double[] { 1, 2, 3, 4 };
            test.Reshape(2, 2);

            Random random = new(1);
            int[] permutation = Enumerable.Range(0, test.Rows)
                                          .OrderBy(k => random.Next())
                                          .ToArray();
            test = test.PermuteRows(permutation);
        }

        [Fact()]
        public void PermuteColumnsTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void ConcatenateColumnsTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void InitRandomTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void SigmoidTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void SumRowsTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void SubtractTest()
        {
            Matrix left = new(2, 2);
            left[0] = 1;
            left[1] = 2;
            left[2] = 3;
            left[3] = 4;

            Matrix right = left;

            Matrix less = left - right;

            less.Sum().Should().Be(0);
        }

        [Fact]
        public void SubtractFromScalarTest()
        {
            Matrix right = new(2, 3);
            right[0] = 1;
            right[1] = 2;
            right[2] = 3;
            right[3] = 4;
            right[4] = 5;
            right[5] = 6;

            double left = 1;

            Matrix less = left - right;

            less.Sum().Should().Be(-15);
        }

        [Fact]
        public void SubtractScalarTest()
        {
            Matrix left = new(2, 3);
            left[0] = 1;
            left[1] = 2;
            left[2] = 3;
            left[3] = 4;
            left[4] = 5;
            left[5] = 6;

            double right = 1;

            Matrix less = left - right;

            less.Sum().Should().Be(15);
        }

        [Fact()]
        public void AddTest()
        {
            Matrix left = new(2, 2);
            left[0] = 1;
            left[1] = 2;
            left[2] = 3;
            left[3] = 4;

            Matrix right = left;

            Matrix add = left + right;

            add.Sum().Should().Be(20);
        }

        [Fact]
        public void MatrixTimesScalarTest()
        {
            Matrix left = new(2, 3);
            left[0] = 1;
            left[1] = 2;
            left[2] = 3;
            left[3] = 4;
            left[4] = 5;
            left[5] = 6;

            double right = 2;

            Matrix times = left * right;

            times.Sum().Should().Be(42);
        }

        [Fact]
        public void ScalarTimesMatrixTest()
        {
            Matrix right = new(2, 3);
            right[0] = 1;
            right[1] = 2;
            right[2] = 3;
            right[3] = 4;
            right[4] = 5;
            right[5] = 6;

            double left = 2;

            Matrix times = left * right;

            times.Sum().Should().Be(42);
        }

        [Fact]
        public void DotTest()
        {
            Matrix left = new(2, 3);
            left[0] = 1;
            left[1] = 2;
            left[2] = 3;
            left[3] = 4;
            left[4] = 5;
            left[5] = 6;

            double[] items = Enumerable.Range(1, 27).Select(i => (double)i).ToArray();
            Matrix right = items;
            right.Reshape(3, 9);

            Matrix dot = left * right;
        }

        [Fact]
        public void DivideMatrixByScalarTest()
        {
            Matrix left = new(2, 3);
            left[0] = 1;
            left[1] = 2;
            left[2] = 3;
            left[3] = 4;
            left[4] = 5;
            left[5] = 6;

            double right = 2;

            Matrix times = left / right;

            times.Sum().Should().Be(10.5);
        }

        [Fact()]
        public void SumColumnsTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void SumTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void MeanTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void VarianceTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void StandardScaleTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void NormalizeTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void UnnormalizeTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void TanhTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void LogTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void LogSumExpTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void SoftmaxTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void PrintTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void RunTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void DisposeTest()
        {
            throw new NotImplementedException();
        }

        [Fact()]
        public void SameShapeTest()
        {
            throw new NotImplementedException();
        }
    }
}