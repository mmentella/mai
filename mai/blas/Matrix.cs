﻿namespace mai.blas
{
    public class Matrix
    {
        private double[,] data;

        public Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;

            data = new double[rows, columns];
        }

        public Matrix(double[,] data)
        {
            Rows = data.GetLength(0);
            Columns = data.GetLength(1);

            this.data = data;
        }

        public Matrix(double[] data)
        {
            Columns = data.Length;
            Rows = 1;

            this.data = new double[Rows, Columns];
            for (int c = 0; c < Columns; c++)
            {
                this.data[0, c] = data[c];
            }
        }

        public int Rows { get; }
        public int Columns { get; }
        public int Length => Rows * Columns;

        public double this[int r, int c]
        {
            get => data[r, c];
            set => data[r, c] = value;
        }

        public Matrix GetRows(int start, int length)
        {
            Matrix rows = new(length, Columns);
            for (int r = start; r < length; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    rows[r,c] = this[r,c];
                }
            }

            return rows;
        }

        public Matrix Transpose()
        {
            Matrix transpose = new(Columns, Rows);

            Run(this, transpose, (l, t, r, c) => t[c, r] = l[r, c]);

            return transpose;
        }

        public Matrix Square()
        {
            Matrix square = new(Rows, Columns);

            Run(this, square, (l, s, r, c) => s[r, c] = l[r, c] * l[r, c]);

            return square;
        }

        public Matrix Hadamard(Matrix matrix)
        {
            Matrix hadamard = new(Rows, Columns);

            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    hadamard[r, c] = this[r, c] * matrix[r, c];
                }
            }

            return hadamard;
        }

        public void PermuteRows(int[] permutation)
        {
            double[,] perm = new double[Rows, Columns];
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    perm[permutation[r], c] = data[r, c];
                }
            }

            Array.Clear(data);
            data = perm;
        }

        public void PermuteColumns(int[] permutation)
        {
            double[,] perm = new double[Rows, Columns];
            for (int c = 0; c < Columns; c++)
            {
                for (int r = 0; r < Rows; r++)
                {
                    perm[r, permutation[c]] = data[r, c];
                }
            }

            Array.Clear(data);
            data = perm;
        }

        public Matrix InitRandom(int? seed = null!)
        {
            Random random = new();
            Run(this, d => d = 2 * random.NextDouble() - 1);

            return this;
        }

        public Matrix Sigmoid()
        {
            Run(this, m => 1 / (1 + Math.Exp(-m)));
            return this;
        }

        public Matrix SumRows()
        {
            double[,] data = new double[1, Columns];
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    data[0, c] += this.data[r, c];
                }
            }

            return new Matrix(data);
        }

        public Matrix SumColumns()
        {
            double[,] data = new double[Rows, 1];
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    data[r, 0] += this.data[r, c];
                }
            }

            return new Matrix(data);
        }

        public double Sum()
        {
            return SumRows().SumColumns()[0, 0];
        }

        public Matrix Tanh()
        {
            Run(this, m => (Math.Exp(m) - Math.Exp(-m)) / (Math.Exp(m) + Math.Exp(-m)));
            return this;
        }

        public string Print() => data.Print();

        public static Matrix operator +(Matrix left, Matrix rigth)
        {
            Matrix add = new(left.Rows, left.Columns);

            for (int r = 0; r < left.Rows; r++)
            {
                for (int c = 0; c < left.Columns; c++)
                {
                    add[r, c] = left[r, c] + rigth[r, c];
                }
            }

            return add;
        }

        public static Matrix operator -(Matrix left, Matrix rigth)
        {
            Matrix less = new(left.Rows, left.Columns);

            for (int r = 0; r < left.Rows; r++)
            {
                for (int c = 0; c < left.Columns; c++)
                {
                    less[r, c] = left[r, c] - rigth[r, c];
                }
            }

            return less;
        }

        public static Matrix operator -(int left, Matrix rigth)
        {
            Matrix less = new(rigth.Rows, rigth.Columns);

            for (int r = 0; r < rigth.Rows; r++)
            {
                for (int c = 0; c < rigth.Columns; c++)
                {
                    less[r, c] = left - rigth[r, c];
                }
            }

            return less;
        }

        public static Matrix operator *(Matrix left, Matrix rigth)
        {
            Matrix dot = new(left.Rows, rigth.Columns);

            for (int r = 0; r < dot.Rows; r++)
            {
                for (int c = 0; c < dot.Columns; c++)
                {
                    for (int k = 0; k < left.Columns; k++)
                    {
                        dot[r, c] += left[r, k] * rigth[k, c];
                    }
                }
            }

            return dot;
        }

        public static Matrix operator /(Matrix left, double right)
        {
            Matrix result = new(left.Rows, left.Columns);

            for (int r = 0; r < left.Rows; r++)
            {
                for (int c = 0; c < left.Columns; c++)
                {
                    result[r, c] = left[r, c] / right;
                }
            }

            return result;
        }

        public static Matrix operator *(Matrix left, double right)
        {
            left.Run(left,l=> l * right);

            return left;
        }

        public static Matrix operator *(double left, Matrix right) => right * left;

        public void Run(Matrix matrix, Func<double, double> func)
        {
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    matrix[r, c] = func(matrix[r, c]);
                }
            }
        }

        public void Run(Matrix left, Matrix rigth, Action<Matrix, Matrix, int, int> action)
        {
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    action(left, rigth, r, c);
                }
            }
        }

        public static Matrix Ones(Matrix matrix)
        {
            Matrix ones = new(matrix.Rows, matrix.Columns);
            ones.Run(ones, d => 1d);

            return ones;
        }

        public static void SameShape(Matrix left, Matrix right)
        {
            if (left.Rows == right.Rows && left.Columns == right.Columns) { return; }

            throw new InvalidOperationException();
        }
    }
}
