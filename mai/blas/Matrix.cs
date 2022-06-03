namespace mai.blas
{
    using FluentAssertions;

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

        public Matrix(double[] data, int rows, int columns)
        {
            data.Length.Should().Be(rows * columns);

            Rows = rows;
            Columns = columns;

            this.data = new double[Rows, Columns];
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    this.data[r, c] = data[r * c + c];
                }
            }
        }

        public int Rows { get; protected set; }
        public int Columns { get; protected set; }
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
                    rows[r, c] = this[r, c];
                }
            }

            return rows;
        }

        public Matrix Transpose()
        {
            Matrix transpose = new(Columns, Rows);

            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    transpose[c, r] = this[r, c];
                }
            }

            return transpose;
        }

        public Matrix Square() => Run(this, d => d * d);

        public Matrix Hadamard(Matrix matrix)
        {
            double[,] hadamard = new double[Rows, Columns];
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    hadamard[r, c] = this[r, c] * matrix[r, c];
                }
            }

            return new(hadamard);
        }

        public Matrix PermuteRows(int[] permutation)
        {
            double[,] perm = new double[Rows, Columns];
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    perm[permutation[r], c] = data[r, c];
                }
            }

            return new(perm);
        }

        public Matrix PermuteColumns(int[] permutation)
        {
            double[,] perm = new double[Rows, Columns];
            for (int c = 0; c < Columns; c++)
            {
                for (int r = 0; r < Rows; r++)
                {
                    perm[r, permutation[c]] = data[r, c];
                }
            }

            return new(perm);
        }

        public Matrix InitRandom(int? seed = null!)
        {
            Random random = seed == null ? new() : new(seed.Value);
            Run(d => d = 2 * random.NextDouble() - 1);

            return this;
        }

        public Matrix Sigmoid() => Run(this, d => 1d / (1d + Math.Exp(-d)));

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

        public double Sum() => SumRows().SumColumns()[0, 0];

        public double Mean() => Sum() / Length;

        public double Variance()
        {
            double mean = Mean();
            double variance = (mean - this).Square().Sum() / Length;

            return variance;
        }

        public double StandardDeviation() => Math.Sqrt(Variance());

        public Matrix StandardScale()
        {
            double mean = Mean();
            Matrix mean0 = mean - this;

            double variance = mean0.Square().Sum() / Length;
            double std = Math.Sqrt(variance);

            return mean0 / std;
        }

        private Matrix Broadcast(int rows)
        {
            double[,] data = new double[rows, Columns];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    data[r, c] = this.data[0, c];
                }
            }

            return new(data);
        }

        public Matrix Tanh() => Run(this, d => (Math.Exp(d) - Math.Exp(-d)) / (Math.Exp(d) + Math.Exp(-d)));

        public Matrix Log() => Run(this, d => Math.Log(d));

        public Matrix Softmax(double? min, double? max)
        {
            min ??= double.MinValue;
            max ??= double.MaxValue;

            double exp;
            double rowSum;
            double[,] softmax = new double[Rows, Columns];
            for (int r = 0; r < Rows; r++)
            {
                rowSum = 0;
                for (int c = 0; c < Columns; c++)
                {
                    exp = Math.Exp(this[r, c]);
                    rowSum += exp;
                    softmax[r, c] = exp;
                }
                for (int c = 0; c < Columns; c++)
                {
                    softmax[r, c] = Math.Min(max.Value, Math.Max(min.Value, softmax[r, c] / rowSum));
                }
            }

            return new(softmax);
        }

        public string Print() => data.Print();

        public static Matrix operator +(Matrix left, Matrix right)
        {
            left.Columns.Should().Be(right.Columns);

            if (left.Rows < right.Rows) { return right + left; }

            if (left.Rows > 1 && right.Rows == 1)
            {
                right = right.Broadcast(left.Rows);
            }

            left.Rows.Should().Be(right.Rows);

            Matrix add = new(left.Rows, left.Columns);

            for (int r = 0; r < left.Rows; r++)
            {
                for (int c = 0; c < left.Columns; c++)
                {
                    add[r, c] = left[r, c] + right[r, c];
                }
            }

            return add;
        }

        public static Matrix operator -(Matrix left, Matrix right)
        {
            Matrix less = new(left.Rows, left.Columns);

            for (int r = 0; r < left.Rows; r++)
            {
                for (int c = 0; c < left.Columns; c++)
                {
                    less[r, c] = left[r, c] - right[r, c];
                }
            }

            return less;
        }

        public static Matrix operator -(double left, Matrix right)
        {
            Matrix less = new(right.Rows, right.Columns);

            for (int r = 0; r < right.Rows; r++)
            {
                for (int c = 0; c < right.Columns; c++)
                {
                    less[r, c] = left - right[r, c];
                }
            }

            return less;
        }

        public static Matrix operator *(Matrix left, Matrix right)
        {
            left.Columns.Should().Be(right.Rows);

            Matrix dot = new(left.Rows, right.Columns);

            for (int r = 0; r < dot.Rows; r++)
            {
                for (int c = 0; c < dot.Columns; c++)
                {
                    for (int k = 0; k < left.Columns; k++)
                    {
                        dot[r, c] += left[r, k] * right[k, c];
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

        public static Matrix operator *(Matrix left, double right) => left.Run(left, l => l * right);

        public static Matrix operator *(double left, Matrix right) => right * left;

        public Matrix Run(Matrix matrix, Func<double, double> func)
        {
            double[,] run = new double[matrix.Rows, matrix.Columns];
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    run[r, c] = func(matrix[r, c]);
                }
            }

            return new(run);
        }

        public void Run(Func<double, double> func)
        {
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    this[r, c] = func(this[r, c]);
                }
            }
        }

        public Matrix Run(Matrix left, Matrix right, Func<Matrix, Matrix, int, int, double> action)
        {
            double[,] run = new double[Rows, Columns];
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    run[r, c] = action(left, right, r, c);
                }
            }

            return new(run);
        }

        public static Matrix Ones(Matrix matrix)
        {
            Matrix ones = new(matrix.Rows, matrix.Columns);
            ones.Run(d => 1d);

            return ones;
        }

        public static void SameShape(Matrix left, Matrix right)
        {
            left.Rows.Should().Be(right.Rows);
            left.Columns.Should().Be(right.Columns);
        }
    }
}
