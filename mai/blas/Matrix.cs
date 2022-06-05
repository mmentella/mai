namespace mai.blas
{
    using FluentAssertions;
    using System.Buffers;
    using System.Diagnostics;
    using System.Runtime.CompilerServices;

    public class Matrix
    {
        private double[] data;

        public Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;

            data = ArrayPool<double>.Shared.Rent(Rows * Columns);
            Array.Fill(data, 0);

            //data = new double[rows * columns];
        }

        public int Rows { get; set; }
        public int Columns { get; set; }
        public int Length => Rows * Columns;

        public double this[int r, int c]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => data[r * Columns + c];

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => data[r * Columns + c] = value;
        }

        public Matrix GetRows(int start, int length)
        {
            Matrix rows = new(length, Columns);

            Parallel.For(start, length, r =>
            {
                Parallel.For(0, Columns, c =>
                {
                    rows[r, c] = this[r, c];
                });
            });

            return rows;
        }

        public Matrix Transpose()
        {
            Matrix transpose = new(Columns, Rows);

            Parallel.For(0, Rows, r =>
            {
                Parallel.For(0, Columns, c =>
                {
                    transpose[c, r] = this[r, c];
                });
            });

            return transpose;
        }

        public Matrix Square() => Run(this, d => d * d);

        public Matrix Hadamard(Matrix matrix)
        {
            Matrix hadamard = new(Rows, Columns);
            Parallel.For(0, Rows, r =>
            {
                Parallel.For(0, Columns, c =>
                {
                    hadamard[r, c] = this[r, c] * matrix[r, c];
                });
            });

            return hadamard;
        }

        public Matrix PermuteRows(int[] permutation)
        {
            Matrix perm = new(Rows, Columns);
            Parallel.For(0, Rows, r =>
            {
                Parallel.For(0, Columns, c =>
                {
                    perm[permutation[r], c] = this[r, c];
                });
            });

            return perm;
        }

        public Matrix PermuteColumns(int[] permutation)
        {
            Matrix perm = new(Rows, Columns);
            Parallel.For(0, Rows, r =>
            {
                Parallel.For(0, Columns, c =>
                {
                    perm[r, permutation[c]] = this[r, c];
                });
            });

            return perm;
        }

        public Matrix ConcatenateColumns(Matrix other)
        {
            Rows.Should().Be(other.Rows);

            int columns = Columns + other.Columns;
            Matrix data = new(Rows, columns);
            Parallel.For(0, Rows, r =>
            {
                Parallel.For(0, Columns, c =>
                {
                    data[r, c] = this[r, c];
                });
                Parallel.For(0, other.Columns, c =>
                {
                    data[r, Columns + c] = other[r, c];
                });
            });

            return data;
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
            Matrix data = new(1, Columns);
            Parallel.For(0, Rows, r =>
            {
                for (int c = 0; c < Columns; c++)
                {
                    data[0, c] += this[r, c];
                }
            });

            return data;
        }

        public Matrix SumColumns()
        {
            Matrix data = new(Rows, 1);
            Parallel.For(0, Columns, c =>
            {
                for (int r = 0; r < Rows; r++)
                {
                    data[r, 0] += this[r, c];
                }
            });

            return data;
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
            Matrix mean0 = this - mean;

            double variance = mean0.Square().Sum() / Length;
            double std = Math.Sqrt(variance);

            return mean0 / std;
        }

        public Matrix Normalize()
        {
            Matrix other = 1 - this;

            return ConcatenateColumns(other);
        }

        public Matrix Unnormalize()
        {
            return GetRows(0, 1);
        }

        private Matrix Broadcast(int rows)
        {
            Matrix data = new(rows, Columns);
            Parallel.For(0, rows, r =>
            {
                Parallel.For(0, Columns, c =>
                {
                    data[r, c] = this[0, c];
                });
            });

            return data;
        }

        public Matrix Tanh() => Run(this, d => (Math.Exp(d) - Math.Exp(-d)) / (Math.Exp(d) + Math.Exp(-d)));

        public Matrix Log() => Run(this, d => Math.Log(d));

        public Matrix LogSumExp()
        {
            double sum;
            Matrix data = new(Rows, 1);
            for (int r = 0; r < Rows; r++)
            {
                sum = 0;
                for (int c = 0; c < Columns; c++)
                {
                    sum += Math.Exp(this[r, c]);
                }
                data[r, 0] = Math.Log(sum);
            }

            return data;
        }

        public Matrix Softmax(double? min = null, double? max = null)
        {
            min ??= double.MinValue;
            max ??= double.MaxValue;

            Matrix logsumexp = LogSumExp();

            Matrix data = new(Rows, Columns);
            Parallel.For(0, Rows, r =>
            {
                Parallel.For(0, Columns, c =>
                {
                    data[r, c] = Math.Min(max.Value, Math.Max(min.Value, Math.Exp(this[r, c] - logsumexp[r, 0])));
                });
            });

            return data;
        }

        public string Print() => data.Print(Rows, Columns);

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
                Parallel.For(0, left.Columns, c => add[r, c] = left[r, c] + right[r, c]);
            }

            return add;
        }

        public static Matrix operator -(Matrix left, Matrix right)
        {
            left.Rows.Should().Be(right.Rows);
            left.Columns.Should().Be(right.Columns);

            Matrix less = new(left.Rows, left.Columns);
            Parallel.For(0, left.Rows, r =>
            {
                Parallel.For(0, left.Columns, c =>
                {
                    less[r, c] = left[r, c] - right[r, c];
                });
            });

            return less;
        }

        public static Matrix operator -(double left, Matrix right)
        {
            Matrix less = new(right.Rows, right.Columns);
            Parallel.For(0, right.Rows, r =>
            {
                Parallel.For(0, right.Columns, c =>
                {
                    less[r, c] = left - right[r, c];
                });
            });

            return less;
        }

        public static Matrix operator -(Matrix left, double right)
        {
            Matrix less = new(left.Rows, left.Columns);
            Parallel.For(0, left.Rows, r =>
            {
                Parallel.For(0, left.Columns, c =>
                {
                    less[r, c] = left[r, c] - right;
                });
            });

            return less;
        }

        public static Matrix operator *(Matrix left, Matrix right)
        {
            left.Columns.Should().Be(right.Rows);

            Matrix dot = new(left.Rows, right.Columns);

            Parallel.For(0, dot.Rows, r =>
            {
                Parallel.For(0, dot.Columns, c =>
                {
                    for (int k = 0; k < left.Columns; k++)
                    {
                        dot[r, c] += left[r, k] * right[k, c];
                    }
                });
            });

            return dot;
        }

        public static Matrix operator /(Matrix left, double right)
        {
            right = 1 / right;
            Matrix result = new(left.Rows, left.Columns);
            Parallel.For(0, left.Rows, r =>
            {
                Parallel.For(0, left.Columns, c =>
                {
                    result[r, c] = left[r, c] * right;
                });
            });

            return result;
        }

        public static Matrix operator *(Matrix left, double right) => left.Run(left, l => l * right);

        public static Matrix operator *(double left, Matrix right) => right * left;

        public Matrix Run(Matrix matrix, Func<double, double> func)
        {
            Matrix run = new(matrix.Rows, matrix.Columns);
            Parallel.For(0, Rows, r =>
            {
                Parallel.For(0, Columns, c => run[r, c] = func(matrix[r, c]));
            });

            return run;
        }

        public void Run(Func<double, double> func)
        {
            Parallel.For(0, Rows, r =>
            {
                Parallel.For(0, Columns, c => this[r, c] = func(this[r, c]));
            });
        }

        public Matrix Run(Matrix left, Matrix right, Func<Matrix, Matrix, int, int, double> action)
        {
            Matrix run = new(Rows, Columns);
            Parallel.For(0, Rows, r =>
            {
                Parallel.For(0, Columns, c => this[r, c] = run[r, c] = action(left, right, r, c));
            });

            return run;
        }

        public void FreeMemory()
        {
            //Debug.WriteLine($"Total Memory Before {GC.GetTotalMemory(true)}");
            ArrayPool<double>.Shared.Return(data);
            //Debug.WriteLine($"Total Memory After {GC.GetTotalMemory(true)}");
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
