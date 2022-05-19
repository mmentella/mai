using System.Linq.Expressions;
using System.Runtime.CompilerServices;

namespace mai.network
{
    public class GatedRecurrentUnit
    {
        private double[] wr;
        private double[] wz;
        private double[] wh;

        private double[] ur;
        private double[] uz;
        private double[] uh;

        private double[] r;
        private double[] z;

        private double[] h;
        private double[] s;

        private double[] o;

        private readonly int inputLength;
        private readonly int stateLength;
        private readonly int memoryLength;

        public GatedRecurrentUnit(int inputLength, int stateLength, int memoryLength)
        {
            ur = new double[memoryLength * inputLength];
            uz = new double[memoryLength * inputLength];
            uh = new double[memoryLength * inputLength];

            wr = new double[memoryLength * memoryLength];
            wz = new double[memoryLength * memoryLength];
            wh = new double[memoryLength * memoryLength];

            r = Array.Empty<double>();
            z = Array.Empty<double>();

            h = new double[stateLength];
            s = new double[stateLength];

            o = new double[stateLength];

            this.inputLength = inputLength;
            this.stateLength = stateLength;
            this.memoryLength = memoryLength;

            RandomInitialization();
        }

        public double[] Forward(double[] input)
        {
            r = Sigmoid(Add(DotProduct(wr, s, stateLength), DotProduct(ur, input, stateLength))).ToArray();
            z = Sigmoid(Add(DotProduct(wz, s, stateLength), DotProduct(uz, input, stateLength))).ToArray();

            h = HyperbolicTanget(Add(DotProduct(wh, Hadamard(r, s), stateLength), DotProduct(uh, input, stateLength))).ToArray();

            s = Add(Hadamard(Less(1, z), h), Hadamard(z, s)).ToArray();

            o = s;

            return o;
        }

        public double[] Train(IList<(double[] sample, double[] label)> trainingSet,
                              double learningRate = 1.0e-1,
                              int epochs = 1000,
                              int k1 = 10,
                              int k2 = 10)
        {
            double[] loss = (double[])Array.CreateInstance(typeof(double), epochs);
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                BackPropagate(trainingSet, learningRate);
            }

            return loss;
        }

        private double CalculateLoss(double[] output, double[] labels)
        {
            return 0.5 * labels.Zip(output, (l, o) => (l - o) * (l - o)).Sum();
        }



        private void BackPropagate(IList<(double[] sample, double[] label)> trainingSet, double learningRate)
        {
            var s0 = s;
            foreach (var (sample, label) in trainingSet)
            {
                IDictionary<int, (double[] z, double[] r, double[] h, double[] s, double[] o)> memory =
                    new Dictionary<int, (double[] z, double[] r, double[] h, double[] s, double[] o)>();
                for (int timestep = 1; timestep <= sample.Length; timestep++)
                {
                    double[] output = Forward(new double[] { sample[timestep - 1] });
                    memory.Add(timestep, (z, r, h, s, o));
                }

                var deltay = Less(memory[sample.Length].o, label);
                double[] dst = (double[])Array.CreateInstance(typeof(double), stateLength);
                double[] dsr = (double[])Array.CreateInstance(typeof(double), stateLength);

                double[] dbh = (double[])Array.CreateInstance(typeof(double), stateLength);
                double[] duh = (double[])Array.CreateInstance(typeof(double), stateLength * inputLength);
                double[] dwh = (double[])Array.CreateInstance(typeof(double), stateLength * stateLength);

                double[] dbr = (double[])Array.CreateInstance(typeof(double), stateLength);
                double[] dur = (double[])Array.CreateInstance(typeof(double), stateLength * inputLength);
                double[] dwr = (double[])Array.CreateInstance(typeof(double), stateLength * stateLength);

                double[] dbz = (double[])Array.CreateInstance(typeof(double), stateLength);
                double[] duz = (double[])Array.CreateInstance(typeof(double), stateLength * inputLength);
                double[] dwz = (double[])Array.CreateInstance(typeof(double), stateLength * stateLength);

                for (int t = sample.Length; t > 1; t--)
                {
                    dst = Add(dst, deltay);
                    var dstCopy = dst.ToArray();
                    var dtanhinput = Hadamard(dst, Less(1, memory[t].z), Less(1, Square(memory[t].h)));
                    
                    dbh = Add(dbh, dtanhinput);
                    duh = Add(duh, DotProduct(dtanhinput, Transpose(new double[] { sample[t - 1] }, inputLength, 1), stateLength, 1, 1, inputLength));
                    dwh = Add(dwh, DotProduct(dtanhinput, Hadamard(memory[t - 1].s, memory[t].r), stateLength, 1, 1, stateLength));

                    dsr = DotProduct(Transpose(wh, stateLength, stateLength), dtanhinput, stateLength, stateLength, stateLength, 1);
                    dst = Hadamard(dsr, memory[t].r);
                    var dsigInputR = Hadamard(dsr, memory[t - 1].s, memory[t].r, Less(1, memory[t].r));
                    
                    dbr = Add(dbr, dsigInputR);
                    dur = Add(duh, DotProduct(dsigInputR, Transpose(new double[] { sample[t - 1] }, inputLength, 1), stateLength, 1, 1, inputLength));
                    dwr = Add(dwh, DotProduct(dsigInputR, memory[t - 1].s, stateLength, 1, 1, stateLength));

                    dst = Add(dst, DotProduct(Transpose(wr, stateLength, stateLength), dsigInputR, stateLength, stateLength, stateLength, 1));
                    dst = Add(dst, Hadamard(dstCopy, memory[t].z));

                    var dz = Hadamard(dstCopy, Less(memory[t - 1].s, memory[t].h));
                    var dsigInputZ = Hadamard(dz, memory[t].z, Less(1, memory[t].z));

                    dbz = Add(dbz, dsigInputZ);
                    duz = Add(duh, DotProduct(dsigInputR, Transpose(new double[] { sample[t - 1] }, inputLength, 1), stateLength, 1, 1, inputLength));
                    dwz = Add(dwh, DotProduct(dsigInputR, memory[t - 1].s, stateLength, 1, 1, stateLength));

                    dst = Add(dst, DotProduct(Transpose(wz, stateLength, stateLength), dsigInputZ, stateLength, stateLength, stateLength, 1));
                }

                dst = Add(dst, deltay);
                var dtanhinput1 = Hadamard(dst, Less(1, memory[1].z), Less(1, Square(memory[1].h)));

                dbh = Add(dbh, dtanhinput1);
                duh = Add(duh, DotProduct(dtanhinput1, Transpose(new double[] { sample[0] }, inputLength, 1), stateLength, 1, 1, inputLength));
                dwh = Add(dwh, DotProduct(dtanhinput1, Hadamard(s0, memory[1].r), stateLength, 1, 1, stateLength));

                dsr = DotProduct(Transpose(wh, stateLength, stateLength), dtanhinput1, stateLength, stateLength, stateLength, 1);
                dst = Hadamard(dsr, memory[1].r);
                var dsigInputR1 = Hadamard(dsr, s0, memory[1].r, Less(1, memory[1].r));

                dbr = Add(dbr, dsigInputR1);
                dur = Add(duh, DotProduct(dsigInputR1, Transpose(new double[] { sample[0] }, inputLength, 1), stateLength, 1, 1, inputLength));
                dwr = Add(dwh, DotProduct(dsigInputR1, s0, stateLength, 1, 1, stateLength));

            }
        }

        private void RandomInitialization(double max = 1.0e-3)
        {
            Random rnd = new();

            ur = ElementWise(ur, u => max * rnd.NextDouble()).ToArray();
            uz = ElementWise(uz, u => max * rnd.NextDouble()).ToArray();
            uh = ElementWise(uh, u => max * rnd.NextDouble()).ToArray();

            wr = ElementWise(wr, w => max * rnd.NextDouble()).ToArray();
            wz = ElementWise(wz, w => max * rnd.NextDouble()).ToArray();
            wh = ElementWise(wh, w => max * rnd.NextDouble()).ToArray();
        }

        public double[] Square(double[] values)
        {
            return ElementWise(values, v => v * v);
        }

        public double[] Transpose(double[] values, int rows, int columns)
        {
            double[] transpose = (double[])Array.CreateInstance(typeof(double), values.Length);

            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < columns; c++)
                {
                    transpose[c * rows + r] = values[r * columns + c];
                }
            }

            return transpose;
        }

        public double[] Sigmoid(double[] values) => ElementWise(values, v => Sigmoid(v));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));

        public double[] SigmoidDerivative(double[] values) =>
            ElementWise(values, v => { double s = Sigmoid(v); return s * (1 - s); });

        public double[] HyperbolicTanget(double[] values) =>
            ElementWise(values, v => HyperbolicTanget(v));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double HyperbolicTanget(double x) =>
            (Math.Exp(x) - Math.Exp(-x)) /
            (Math.Exp(x) + Math.Exp(-x));

        public double[] HyperbolicTangetDerivative(double[] values) =>
            ElementWise(values, v => { double h = HyperbolicTanget(v); return 1 - h * h; });

        public double[] Add(double[] left, double[] right)
        {
            double[] sum = (double[])Array.CreateInstance(typeof(double), left.Length);
            for (int i = 0; i < left.Length; i++)
            {
                sum[i] = left[i] + right[i];
            }

            return sum;
        }

        public double[] Add(params double[][] values)
        {
            double[] sum = (double[])Array.CreateInstance(typeof(double), values[0].Length);
            for (int i = 0; i < sum.Length; i++)
            {
                sum[i] = 0;
                for (int j = 0; j < values.Length; j++)
                {
                    sum[i] += values[j][i];
                }
            }

            return sum;
        }

        public double[] Less(double[] left, double[] right)
        {
            double[] sum = (double[])Array.CreateInstance(typeof(double), left.Length);
            for (int i = 0; i < left.Length; i++)
            {
                sum[i] = left[i] - right[i];
            }

            return sum;
        }

        public double[] ElementWise(double[] left, double[] right, Func<double, double, double> func)
        {
            double[] result = (double[])Array.CreateInstance(typeof(double), left.Length);
            for (int i = 0; i < left.Length; i++)
            {
                result[i] = func(left[i], right[i]);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double[] ElementWise(double[] values, Func<double, double> func)
        {
            double[] result = (double[])Array.CreateInstance(typeof(double), values.Length);
            for (int i = 0; i < values.Length; i++)
            {
                result[i] = func(values[i]);
            }

            return result;
        }

        public double[] Less(double left, double[] right)
        {
            double[] less = (double[])Array.CreateInstance(typeof(double), right.Length);
            for (int i = 0; i < right.Length; i++)
            {
                less[i] = left - right[i];
            }

            return less;
        }

        public double[] Hadamard(double[] left, double[] right)
        {
            double[] hadamard = (double[])Array.CreateInstance(typeof(double), left.Length);
            for (int i = 0; i < left.Length; i++)
            {
                hadamard[i] = left[i] * right[i];
            }

            return hadamard;
        }

        public double[] Hadamard(params double[][] values)
        {
            double[] hadamard = (double[])Array.CreateInstance(typeof(double), values[0].Length);
            for (int i = 0; i < hadamard.Length; i++)
            {
                hadamard[i] = 1;
                for (int j = 0; j < values.Length; j++)
                {
                    hadamard[i] *= values[j][i];
                }
            }

            return hadamard;
        }

        public double[] DotProduct(double[] matrix, double[] vector, int rows)
        {
            double[] dot = (double[])Array.CreateInstance(typeof(double), rows);
            for (int i = 0; i < rows; i++)
            {
                dot[i] = Reduce(matrix[(i * vector.Length)..((i + 1) * vector.Length)], vector);
            }

            return dot;
        }

        public double[] DotProduct(double[] left, double[] right, params int[] dim)
        {
            double[] dot = (double[])Array.CreateInstance(typeof(double), stateLength * dim[3]);

            right = Transpose(right, dim[2], dim[3]);
            for (int r = 0; r < stateLength; r++)
            {
                for (int c = 0; c < dim[3]; c++)
                {
                    dot[r * dim[3] + c] = Reduce(left[(r * inputLength)..((r + 1) * inputLength)], right[(c * dim[2])..((c + 1) * dim[2])]);
                }
            }

            return dot;
        }

        public double Reduce(double[] left, double[] right)
        {
            double reduce = 0;
            for (int i = 0; i < left.Length; i++)
            {
                reduce += left[i] * right[i];
            }

            return reduce;
        }

        public double[] Softmax(double[] vector)
        {
            double energy = 0;
            double[] softmax = (double[])Array.CreateInstance(typeof(double), vector.Length);
            for (int i = 0; i < softmax.Length; i++)
            {
                softmax[i] = Math.Exp(softmax[i]);
                energy += softmax[i];
            }

            energy = 1 / energy;

            for (int i = 0; i < softmax.Length; i++)
            {
                softmax[i] = energy * softmax[i];
            }

            return softmax;
        }

        public double[] Zeros(params int[] size)
        {
            int length = 1;
            for (int i = 0; i < size.Length; i++) { length *= i; }

            return (double[])Array.CreateInstance(typeof(double), length);
        }

        public double[] Ones(params int[] size)
        {
            int length = 1;
            for (int i = 0; i < size.Length; i++) { length *= i; }

            double[] ones = (double[])Array.CreateInstance(typeof(double), length);
            Array.Fill(ones, 1d);

            return ones;
        }
    }
}
