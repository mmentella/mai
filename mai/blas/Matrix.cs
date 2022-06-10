namespace mai.blas
{
    using FluentAssertions;
    using System.Diagnostics;
    using System.Runtime.CompilerServices;
    using System.Runtime.InteropServices;

    public class Matrix
        : IDisposable
    {
        protected IntPtr data;
        protected readonly int sizeofFLoat = sizeof(float);

        public unsafe Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;

            data = Marshal.AllocHGlobal(sizeof(float) * Length);
            Unsafe.InitBlock(data.ToPointer(), 0, (uint)(sizeof(float) * Length));
            //Debug.WriteLine($"{GetHashCode()} Created");
        }

        public int Rows { get; protected set; }
        public int Columns { get; protected set; }
        public int Length => Rows * Columns;

        public void Reshape(int rows, int columns)
        {
            Length.Should().Be(rows * columns);

            Rows = rows;
            Columns = columns;
        }

        public float this[int r, int c]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => this[r * Columns + c];

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => this[r * Columns + c] = value;
        }

        public unsafe float this[int i]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ((float*)this)[i];

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => ((float*)this)[i] = value;
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

            for (int c = 0; c < Columns; c++)
            {
                for (int r = 0; r < Rows; r++)
                {
                    transpose[c, r] = this[r, c];
                }
            }

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
                this[l] = (float)(2 * random.NextDouble() - 1);
            }

            return this;
        }

        public Matrix Sigmoid() => Run(this, d => 1f / (1f + (float)Math.Exp(-d)));

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

        public float Sum() => SumRows().SumColumns()[0, 0];

        public float Mean() => Sum() / Length;

        public float Variance()
        {
            float mean = Mean();
            float variance = (mean - this).Square().Sum() / Length;

            return variance;
        }

        public Matrix StandardScale()
        {
            float mean = Mean();
            Matrix mean0 = this - mean;

            float variance = mean0.Square().Sum() / Length;
            float std = (float)Math.Sqrt(variance);

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
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    data[r, c] = this[0, c];
                }
            }

            return data;
        }

        public Matrix Tanh() => Run(this, d => (float)((Math.Exp(d) - Math.Exp(-d)) / (Math.Exp(d) + Math.Exp(-d))));

        public Matrix Log() => Run(this, d => (float)Math.Log(d));

        public Matrix LogSumExp()
        {
            float sum;
            Matrix data = new(Rows, 1);
            for (int r = 0; r < Rows; r++)
            {
                sum = 0;
                for (int c = 0; c < Columns; c++)
                {
                    sum += (float)Math.Exp(this[r, c]);
                }
                data[r, 0] = (float)Math.Log(sum);
            }

            return data;
        }

        public Matrix Softmax(float? min = null, float? max = null)
        {
            min ??= float.MinValue;
            max ??= float.MaxValue;

            Matrix logsumexp = LogSumExp();

            Matrix data = new(Rows, Columns);
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    data[r, c] = Math.Min(max.Value, Math.Max(min.Value, (float)Math.Exp(this[r, c] - logsumexp[r, 0])));
                }
            }

            return data;
        }

        public string Print() => this.Print(Rows, Columns);

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

            for (int l = 0; l < left.Length; l++)
            {
                add[l] = left[l] + right[l];
            }

            return add;
        }

        public static Matrix operator -(Matrix left, Matrix right)
        {
            left.Rows.Should().Be(right.Rows);
            left.Columns.Should().Be(right.Columns);

            Matrix less = new(left.Rows, left.Columns);
            for (int s = 0; s < less.Length; s++)
            {
                less[s] = left[s] - right[s];
            }

            return less;
        }

        public static Matrix operator -(float left, Matrix right)
        {
            Matrix less = new(right.Rows, right.Columns);
            for (int l = 0; l < right.Length; l++)
            {
                less[l] = left - right[l];
            }

            return less;
        }

        public static Matrix operator -(Matrix left, float right)
        {
            Matrix less = new(left.Rows, left.Columns);
            for (int l = 0; l < left.Length; l++)
            {
                less[l] = left[l] - right;
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

        public static Matrix operator /(Matrix left, float right)
        {
            right = 1 / right;
            Matrix result = new(left.Rows, left.Columns);
            for (int r = 0; r < left.Rows; r++)
            {
                for (int c = 0; c < left.Columns; c++)
                {
                    result[r, c] = left[r, c] * right;
                }
            }

            return result;
        }

        public static Matrix operator *(Matrix left, float right) => left.Run(left, l => l * right);

        public static Matrix operator *(float left, Matrix right) => right * left;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe implicit operator float*(in Matrix matrix) =>
            matrix.data != IntPtr.Zero ?
            (float*)matrix.data.ToPointer() :
            throw new InvalidOperationException();

        public Matrix Run(Matrix matrix, Func<float, float> func)
        {
            Matrix run = new(matrix.Rows, matrix.Columns);

            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    run[r, c] = func(matrix[r, c]);
                }
            }

            return run;
        }

        public void Dispose()
        {
            Marshal.FreeHGlobal(data);
        }

        public static void SameShape(Matrix left, Matrix right)
        {
            left.Rows.Should().Be(right.Rows);
            left.Columns.Should().Be(right.Columns);
        }
    }
}
