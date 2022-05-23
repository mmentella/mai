namespace mai.blas
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
            Rows = data.Length;
            Columns = 1;

            this.data = new double[Rows, Columns];
            for (int r = 0; r < Rows; r++)
            {
                this.data[r, 0] = data[r];
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

        public Matrix InitRandom()
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

        private void Run(Matrix matrix, Func<double, double> func)
        {
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    matrix[r, c] = func(matrix[r, c]);
                }
            }
        }

        private void Run(Matrix left, Matrix rigth, Action<Matrix, Matrix, int, int> action)
        {
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Columns; c++)
                {
                    action(left, rigth, r, c);
                }
            }
        }
    }
}
