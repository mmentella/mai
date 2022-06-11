namespace mai.blas
{
    using FluentAssertions;
    using System.Diagnostics;
    using System.Numerics;
    using System.Runtime.CompilerServices;
    using System.Runtime.InteropServices;

    public class Matrix
        : IDisposable
    {
        protected double[] data;

        public unsafe Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;

            data = new double[rows * columns];
            //data = Marshal.AllocHGlobal(sizeof(double) * Length);
            //Unsafe.InitBlock(data.ToPointer(), 0, (uint)(sizeof(double) * Length));
            //Debug.WriteLine($"{GetHashCode()} Created");
        }

        public int Rows { get; protected set; }
        public int Columns { get; protected set; }
        public int Length => Rows * Columns;

        public Matrix Reshape(int rows, int columns)
        {
            Length.Should().Be(rows * columns);

            Rows = rows;
            Columns = columns;

            return this;
        }

        public double this[int r, int c]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => this[r * Columns + c];

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => this[r * Columns + c] = value;
        }

        public unsafe double this[int i]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => data[i];

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => data[i] = value;
        }

        public Matrix GetRows(int start, int length)
        {
            (start + length).Should().BeLessThanOrEqualTo(Rows);

            Matrix rows = new(length, Columns);

            for (int r = start; r < start + length; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    rows[r - start, c] = this[r, c];
                }
            }

            return rows;
        }

        public Matrix Transpose()
        {
            Matrix transpose = new(Columns, Rows);

            void Kernel(int l)
            {
                if (l == Length - 1) 
                {
                    if(l == 0 || l == Length - 1) { transpose[l] = this[l]; }
                    return;
                }
                transpose[(l * Rows) % (Length - 1)] = this[l];
            }
            ParallelLoopResult plr = Parallel.For(0, Length, Kernel);
            plr.IsCompleted.Should().BeTrue();

            return transpose;
        }

        public Matrix Square() => Run(this, d => d * d);

        public Matrix Hadamard(Matrix matrix)
        {
            matrix.Rows.Should().Be(Rows);
            matrix.Columns.Should().Be(Columns);

            Matrix hadamard = new(Rows, Columns);

            for (int c = 0; c < Columns; c++)
            {
                for (int r = 0; r < Rows; r++)
                {
                    hadamard[r, c] = this[r, c] * matrix[r, c];
                }
            }

            return hadamard;
        }

        public Matrix PermuteRows(int[] permutation)
        {
            Matrix perm = new(Rows, Columns);
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    perm[permutation[r], c] = this[r, c];
                }
            }

            return perm;
        }

        public Matrix PermuteColumns(int[] permutation)
        {
            Matrix perm = new(Rows, Columns);
            for (int c = 0; c < Columns; c++)
            {
                for (int r = 0; r < Rows; r++)
                {
                    perm[r, permutation[c]] = this[r, c];
                }
            }

            return perm;
        }

        public Matrix ConcatenateColumns(Matrix other)
        {
            Rows.Should().Be(other.Rows);

            int columns = Columns + other.Columns;
            Matrix data = new(Rows, columns);
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    data[r, c] = this[r, c];
                }
                for (int c = 0; c < Columns; c++)
                {
                    data[r, Columns + c] = other[r, c];
                }
            }

            return data;
        }

        public Matrix InitRandom(int? seed = null!)
        {
            Random random = seed == null ? new() : new(seed.Value);
            for (int l = 0; l < Length; l++)
            {
                this[l] = (double)(2 * random.NextDouble() - 1);
            }

            return this;
        }

        public Matrix Sigmoid() => Run(this, d => 1f / (1f + (double)Math.Exp(-d)));

        public Matrix SumRows()
        {
            Matrix data = new(1, Columns);
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    data[0, c] += this[r, c];
                }
            }

            return data;
        }

        public Matrix SumColumns()
        {
            Matrix data = new(Rows, 1);
            for (int c = 0; c < Columns; c++)
            {
                for (int r = 0; r < Rows; r++)
                {
                    data[r, 0] += this[r, c];
                }
            }

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

        public Matrix StandardScale()
        {
            double mean = Mean();
            Matrix mean0 = this - mean;

            double variance = mean0.Square().Sum() / Length;
            double std = (double)Math.Sqrt(variance);

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
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    data[r, c] = this[0, c];
                }
            }

            return data;
        }

        public Matrix Tanh() => Run(this, d => (double)((Math.Exp(d) - Math.Exp(-d)) / (Math.Exp(d) + Math.Exp(-d))));

        public Matrix Log() => Run(this, d => (double)Math.Log(d));

        public Matrix LogSumExp()
        {
            double sum;
            Matrix data = new(Rows, 1);
            for (int r = 0; r < Rows; r++)
            {
                sum = 0;
                for (int c = 0; c < Columns; c++)
                {
                    sum += (double)Math.Exp(this[r, c]);
                }
                data[r, 0] = (double)Math.Log(sum);
            }

            return data;
        }

        public Matrix Softmax(double? min = null, double? max = null)
        {
            min ??= double.MinValue;
            max ??= double.MaxValue;

            Matrix logsumexp = LogSumExp();

            Matrix data = new(Rows, Columns);
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    data[r, c] = Math.Min(max.Value, Math.Max(min.Value, (double)Math.Exp(this[r, c] - logsumexp[r, 0])));
                }
            }

            return data;
        }

        public string Print() => this.Print(Rows, Columns);

        public static Matrix operator +(Matrix left, Matrix right)
        {
            left.Columns.Should().Be(right.Columns);

            if (left.Rows < right.Rows) { return right + left; }

            Matrix right_ = right;
            if (left.Rows > 1 && right.Rows == 1)
            {
                right_ = right.Broadcast(left.Rows);
            }

            left.Rows.Should().Be(right_.Rows);

            Span<Vector<double>> lspan = MemoryMarshal.Cast<double, Vector<double>>(left);
            Span<Vector<double>> rspan = MemoryMarshal.Cast<double, Vector<double>>(right_);

            Matrix add = new(left.Rows, left.Columns);
            for (int s = 0; s < lspan.Length; s++)
            {
                Vector.Add(lspan[s], rspan[s]).CopyTo(add, s * Vector<double>.Count);
            }

            for (int s = lspan.Length * Vector<double>.Count; s < add.Length; s++)
            {
                add[s] = left[s] + right_[s];
            }

            return add;
        }

        public static Matrix operator -(Matrix left, Matrix right)
        {
            left.Rows.Should().Be(right.Rows);
            left.Columns.Should().Be(right.Columns);

            Span<Vector<double>> lspan = MemoryMarshal.Cast<double, Vector<double>>(left);
            Span<Vector<double>> rspan = MemoryMarshal.Cast<double, Vector<double>>(right);

            Matrix less = new(left.Rows, left.Columns);
            for (int s = 0; s < lspan.Length; s++)
            {
                Vector.Subtract(lspan[s], rspan[s]).CopyTo(less, s * Vector<double>.Count);
            }

            for (int s = lspan.Length * Vector<double>.Count; s < less.Length; s++)
            {
                less[s] = left[s] - right[s];
            }

            return less;
        }

        public static Matrix operator -(double left, Matrix right)
        {
            double[] vleft = Enumerable.Repeat(left, Vector<double>.Count).ToArray();

            Span<Vector<double>> lspan = MemoryMarshal.Cast<double, Vector<double>>(vleft);
            Span<Vector<double>> rspan = MemoryMarshal.Cast<double, Vector<double>>(right);

            Matrix less = new(right.Rows, right.Columns);
            for (int s = 0; s < rspan.Length; s++)
            {
                Vector.Subtract(lspan[0], rspan[s]).CopyTo(less, s * Vector<double>.Count);
            }

            for (int s = rspan.Length * Vector<double>.Count; s < right.Length; s++)
            {
                less[s] = left - right[s];
            }

            return less;
        }

        public static Matrix operator -(Matrix left, double right)
        {
            double[] vright = Enumerable.Repeat(right, Vector<double>.Count).ToArray();

            Span<Vector<double>> lspan = MemoryMarshal.Cast<double, Vector<double>>(left);
            Span<Vector<double>> rspan = MemoryMarshal.Cast<double, Vector<double>>(vright);

            Matrix less = new(left.Rows, left.Columns);
            for (int s = 0; s < lspan.Length; s++)
            {
                Vector.Subtract(lspan[s], rspan[0]).CopyTo(less, s * Vector<double>.Count);
            }

            for (int s = lspan.Length * Vector<double>.Count; s < left.Length; s++)
            {
                less[s] = left[s] - right;
            }

            return less;
        }

        public static Matrix operator *(Matrix left, Matrix right)
        {
            left.Columns.Should().Be(right.Rows);

            Matrix dot = new(left.Rows, right.Columns);

            right = right.Transpose();

            Span<double> lspan = left;
            Span<double> rspan = right;
            for (int r = 0; r < dot.Rows; r++)
            {
                Span<Vector<double>> lsv = MemoryMarshal.Cast<double, Vector<double>>(lspan.Slice(r * left.Columns, left.Columns));
                for (int c = 0; c < dot.Columns; c++)
                {
                    Span<Vector<double>> rsv = MemoryMarshal.Cast<double, Vector<double>>(rspan.Slice(c * right.Columns, right.Columns));
                    for (int s = 0; s < lsv.Length; s++)
                    {
                        dot[r, c] = Vector.Dot(lsv[s], rsv[s]);
                    }

                    for (int s = lsv.Length * Vector<double>.Count; s < left.Columns; s++)
                    {
                        dot[r, c] += left[r, s] * right[c, s];
                    }
                }
            }

            return dot;
        }

        public static Matrix operator /(Matrix left, double right)
        {
            right = 1 / right;

            return left * right;
        }

        public static Matrix operator *(Matrix left, double right)
        {
            double[] vright = Enumerable.Repeat(right, Vector<double>.Count).ToArray();

            Span<Vector<double>> lspan = MemoryMarshal.Cast<double, Vector<double>>(left);
            Span<Vector<double>> rspan = MemoryMarshal.Cast<double, Vector<double>>(vright);

            Matrix times = new(left.Rows, left.Columns);
            for (int s = 0; s < lspan.Length; s++)
            {
                Vector.Multiply(lspan[s], rspan[0]).CopyTo(times, s * Vector<double>.Count);
            }

            for (int s = lspan.Length * Vector<double>.Count; s < left.Length; s++)
            {
                times[s] = left[s] * right;
            }

            return times;
        }

        public static Matrix operator *(double left, Matrix right) => right * left;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator double[](in Matrix matrix) => matrix.data;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator Span<double>(in Matrix matrix) => matrix.data;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator Matrix(in double[] data) => new(1, data.Length) { data = data };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Matrix Run(Matrix matrix, Func<double, double> func)
        {
            Matrix run = new(matrix.Rows, matrix.Columns);

            void Kernel(int l)
            {
                run[l] = func(matrix[l]);
            }
            ParallelLoopResult plr = Parallel.For(0, Length, Kernel);
            plr.IsCompleted.Should().BeTrue();

            return run;
        }

        public void Dispose()
        {

        }

        public static void SameShape(Matrix left, Matrix right)
        {
            left.Rows.Should().Be(right.Rows);
            left.Columns.Should().Be(right.Columns);
        }
    }
}
