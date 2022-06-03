﻿using System.Diagnostics;
using Xunit;

namespace mai.blas.Tests
{
    public class MatrixTests
    {
        [Fact()]
        public void TransposeTest()
        {
            Matrix zero = new(3, 5);
            Matrix transpose = zero.Transpose();
        }

        [Fact()]
        public void MatrixTest()
        {
            Matrix matrix = new(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            Matrix transpose = matrix.Transpose();

            Debug.WriteLine(matrix.Print());
            Debug.WriteLine(transpose.Print());
        }

        [Fact()]
        public void PrintTest()
        {
            Matrix one = new(new double[,] { { 1 } });
            Matrix matrix = new(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            Matrix transpose = matrix.Transpose();
            Matrix square = matrix.Square();
            Matrix hadamard = matrix.Hadamard(matrix);

            Debug.WriteLine(one.Print());
            Debug.WriteLine(matrix.Print());
            Debug.WriteLine(transpose.Print());
            Debug.WriteLine(square.Print());
            Debug.WriteLine(hadamard.Print());
        }
    }
}