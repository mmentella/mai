using System.Linq.Expressions;

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

        private int[] dim;

        private double dur;
        private double duz;
        private double duh;
        private double dwr;
        private double dwz;
        private double dwh;

        public GatedRecurrentUnit(int inputLength, int stateLength)
        {
            ur = new double[stateLength * inputLength];
            uz = new double[stateLength * inputLength];
            uh = new double[stateLength * inputLength];

            wr = new double[stateLength * stateLength];
            wz = new double[stateLength * stateLength];
            wh = new double[stateLength * stateLength];

            r = Array.Empty<double>();
            z = Array.Empty<double>();

            h = new double[stateLength];
            s = new double[stateLength];

            o = new double[stateLength];

            dim = new int[] { stateLength, inputLength };

            RandomInitialization();
        }

        public double[] Forward(double[] input)
        {
            r = Sigmoid(Add(DotProduct(wr, s, dim[0]), DotProduct(ur, input, dim[0]))).ToArray();
            z = Sigmoid(Add(DotProduct(wz, s, dim[0]), DotProduct(uz, input, dim[0]))).ToArray();

            h = HyperbolicTanget(Add(DotProduct(wh, Hadamard(r, s), dim[0]), DotProduct(uh, input, dim[0]))).ToArray();

            s = Add(Hadamard(Less(1, z), h), Hadamard(z, s)).ToArray();

            o = s;

            return o;
        }

        public double[] Train(IList<(double[] sample, double label)> trainingSet,
                              double learningRate = 1.0e-1,
                              int epochs = 1000,
                              int k1 = 10,
                              int k2 = 10)
        {
            double[] loss = (double[])Array.CreateInstance(typeof(double), epochs);
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                BackPropagate(trainingSet, learningRate, k1, k2);
            }

            return loss;
        }

        private double CalculateLoss(double[] output, double[] labels)
        {
            return 0.5 * labels.Zip(output, (l, o) => (l - o) * (l - o)).Sum();
        }



        private void BackPropagate(IList<(double[] sample, double label)> trainingSet,
                                   double learningRate,
                                   int k1,
                                   int k2)
        {
            Span<double> dldbz = (double[])Array.CreateInstance(typeof(double), dim[0]);
            Span<double> dldwz = (double[])Array.CreateInstance(typeof(double), dim[0] * dim[0]);
            Span<double> dlduz = (double[])Array.CreateInstance(typeof(double), dim[0] * dim[1]);
                          
            Span<double> dldbr = (double[])Array.CreateInstance(typeof(double), dim[0]);
            Span<double> dldwr = (double[])Array.CreateInstance(typeof(double), dim[0] * dim[0]);
            Span<double> dldur = (double[])Array.CreateInstance(typeof(double), dim[0] * dim[1]);
                          
            Span<double> dldbh = (double[])Array.CreateInstance(typeof(double), dim[0]);
            Span<double> dldwh = (double[])Array.CreateInstance(typeof(double), dim[0] * dim[0]);
            Span<double> dlduh = (double[])Array.CreateInstance(typeof(double), dim[0] * dim[1]);
            /***********************************************************************************/
            Span<double> dsdbz = (double[])Array.CreateInstance(typeof(double), dim[0]);
            Span<double> dsdwz = (double[])Array.CreateInstance(typeof(double), dim[0] * dim[0]);
            Span<double> dsduz = (double[])Array.CreateInstance(typeof(double), dim[0] * dim[1]);

            Span<double> dsdbr = (double[])Array.CreateInstance(typeof(double), dim[0]);
            Span<double> dsdwr = (double[])Array.CreateInstance(typeof(double), dim[0] * dim[0]);
            Span<double> dsdur = (double[])Array.CreateInstance(typeof(double), dim[0] * dim[1]);

            Span<double> dsdbh = (double[])Array.CreateInstance(typeof(double), dim[0]);
            Span<double> dsdwh = (double[])Array.CreateInstance(typeof(double), dim[0] * dim[0]);
            Span<double> dsduh = (double[])Array.CreateInstance(typeof(double), dim[0] * dim[1]);

            Span<double> dsds = (double[])Array.CreateInstance(typeof(double), dim[0]);

            foreach (var training in trainingSet)
            {
                double[] input = training.sample;

                double[] sprev = (double[])Array.CreateInstance(typeof(double), s.Length);
                Array.Copy(s, sprev, s.Length);

                double[] output = Forward(input);

                var dlds = Less(output, s);

                dsdbz = Hadamard(Less(sprev, h).ToArray(), z, Less(1, z).ToArray());
                dsdwz = DotProduct(dsdbz, sprev, dim[0], 1, 1, dim[0]);
                dsduz = DotProduct(dsdbz, input, dim[0], 1, 1, dim[1]);

                dsdbr = Hadamard(Less(1, z).ToArray(),
                                 DotProduct(Transpose(wh, dim[0], dim[0]),
                                            Hadamard(sprev, Less(1, Square(h))), dim[0])
                                 .ToArray(),
                                 r,
                                 Less(1, r).ToArray());
                dsdwr = DotProduct(dsdbr, sprev, dim[0], 1, 1, dim[0]);
                dsdur = DotProduct(dsdbr, input, dim[0], 1, 1, dim[1]);

                dsdbh = Hadamard(Less(1, z), Less(1, Square(h)));
                dsdwh = DotProduct(dsdbh, Hadamard(r, sprev), dim[0], 1, 1, dim[0]);
                dsduh = DotProduct(dsdbh, input, dim[0], 1, 1, dim[1]);

                var dsidsj = 
                    Add(z,
                        Hadamard(Less(sprev, h),
                                        DotProduct(Transpose(wz, dim[0], dim[0]),
                                                   Hadamard(z, Less(1, z)), dim[0])).ToArray(),
                        Hadamard(Less(1, z),
                                 Add(Hadamard(DotProduct(Transpose(wh, dim[0], dim[0]),
                                                         Hadamard(sprev, Less(1, Square(h))), dim[0]),
                                              DotProduct(Transpose(wr, dim[0], dim[0]),
                                                         Hadamard(r, Less(1, r)), dim[0])),
                                     DotProduct(Transpose(wh, dim[0], dim[0]),
                                                Hadamard(r, Less(1, Square(h))), dim[0]))).ToArray());
                dsds = Hadamard(dsds, dsidsj);
                dldbz = Add(dldbz, Hadamard(dlds.ToArray(), dsds.ToArray(), dsdbz.ToArray()));
                dldwz = Add(dldwz, DotProduct(DotProduct(dlds, dsds, dim[0], 1, 1, dim[0]), 
                                              dsdwz, dim[0], dim[0], dim[0], dim[0]));
                dlduz = Add(dlduz, DotProduct(DotProduct(dlds, dsds, dim[0], 1, 1, dim[0]), 
                                              dsduz, dim[0], dim[0], dim[0], dim[1]));

                dldbr = Add(dldbr, Hadamard(dlds.ToArray(), dsds.ToArray(), dsdbr.ToArray()));
                dldwr = Add(dldwr, DotProduct(DotProduct(dlds, dsds, dim[0], 1, 1, dim[0]), 
                                              dsdwr, dim[0], dim[0], dim[0], dim[0]));
                dldur = Add(dldur, DotProduct(DotProduct(dlds, dsds, dim[0], 1, 1, dim[0]), 
                                              dsdur, dim[0], dim[0], dim[0], dim[1]));

                dldbh = Add(dldbh, Hadamard(dlds.ToArray(), dsds.ToArray(), dsdbh.ToArray()));
                dldwh = Add(dldwh, DotProduct(DotProduct(dlds, dsds, dim[0], 1, 1, dim[0]), 
                                              dsdwh, dim[0], dim[0], dim[0], dim[0]));
                dlduh = Add(dlduh, DotProduct(DotProduct(dlds, dsds, dim[0], 1, 1, dim[0]),
                                              dsduh, dim[0], dim[0], dim[0], dim[1]));
            }

        }

        private void UpdateWeights(double learningRate)
        {
            ur = ElementWise(ur, u => u - learningRate * dur).ToArray();
            uz = ElementWise(uz, u => u - learningRate * duz).ToArray();
            uh = ElementWise(uh, u => u - learningRate * duh).ToArray();

            wr = ElementWise(wr, w => w - learningRate * dwr).ToArray();
            wz = ElementWise(wz, w => w - learningRate * dwz).ToArray();
            wh = ElementWise(wh, w => w - learningRate * dwh).ToArray();
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

        public Span<double> Square(Span<double> values)
        {
            return ElementWise(values, v => v * v);
        }

        public Span<double> Transpose(Span<double> values, int rows, int columns)
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

        public Span<double> Sigmoid(Span<double> values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = 1 / (1 + Math.Exp(-values[i]));
            }

            return values;
        }

        public Span<double> SigmoidDerivative(Span<double> values)
        {
            var sigmoid = Sigmoid(values);

            return Hadamard(sigmoid, Less(1, sigmoid));
        }

        public Span<double> HyperbolicTanget(Span<double> values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (Math.Exp(values[i]) - Math.Exp(-values[i])) /
                            (Math.Exp(values[i]) + Math.Exp(-values[i]));
            }

            return values;
        }

        public Span<double> HyperbolicTangetDerivative(Span<double> values)
        {
            var tanh = HyperbolicTanget(values);
            tanh = ElementWise(tanh, v => v * v);

            return Less(1, tanh);
        }

        public Span<double> Add(Span<double> left, Span<double> right)
        {
            double[] sum = (double[])Array.CreateInstance(typeof(double), left.Length);
            for (int i = 0; i < left.Length; i++)
            {
                sum[i] = left[i] + right[i];
            }

            return sum;
        }

        public Span<double> Add(params double[][] values)
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

        public Span<double> Less(Span<double> left, Span<double> right)
        {
            double[] sum = (double[])Array.CreateInstance(typeof(double), left.Length);
            for (int i = 0; i < left.Length; i++)
            {
                sum[i] = left[i] - right[i];
            }

            return sum;
        }

        public Span<double> ElementWise(Span<double> left, Span<double> right, Func<double, double, double> func)
        {
            double[] result = (double[])Array.CreateInstance(typeof(double), left.Length);
            for (int i = 0; i < left.Length; i++)
            {
                result[i] = func(left[i], right[i]);
            }

            return result;
        }

        public Span<double> ElementWise(Span<double> values, Func<double, double> func)
        {
            double[] result = (double[])Array.CreateInstance(typeof(double), values.Length);
            for (int i = 0; i < values.Length; i++)
            {
                result[i] = func(values[i]);
            }

            return result;
        }

        public Span<double> Less(double left, Span<double> right)
        {
            double[] less = (double[])Array.CreateInstance(typeof(double), right.Length);
            for (int i = 0; i < right.Length; i++)
            {
                less[i] = left - right[i];
            }

            return less;
        }

        public Span<double> Hadamard(Span<double> left, Span<double> right)
        {
            double[] hadamard = (double[])Array.CreateInstance(typeof(double), left.Length);
            for (int i = 0; i < left.Length; i++)
            {
                hadamard[i] = left[i] * right[i];
            }

            return hadamard;
        }

        public Span<double> Hadamard(params double[][] values)
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

        public Span<double> DotProduct(Span<double> matrix, Span<double> vector, int rows)
        {
            double[] dot = (double[])Array.CreateInstance(typeof(double), rows);
            for (int i = 0; i < rows; i++)
            {
                dot[i] = Reduce(matrix.Slice(i * vector.Length, vector.Length), vector);
            }

            return dot;
        }

        public Span<double> DotProduct(Span<double> left, Span<double> right, params int[] dim)
        {
            double[] dot = (double[])Array.CreateInstance(typeof(double), dim[0] * dim[3]);

            right = Transpose(right, dim[2], dim[3]);
            for (int r = 0; r < dim[0]; r++)
            {
                for (int c = 0; c < dim[3]; c++)
                {
                    dot[r * dim[3] + c] = Reduce(left.Slice(r * dim[1], dim[1]), right.Slice(c * dim[2], dim[2]));
                }
            }

            return dot;
        }

        public double Reduce(Span<double> left, Span<double> right)
        {
            double reduce = 0;
            for (int i = 0; i < left.Length; i++)
            {
                reduce += left[i] * right[i];
            }

            return reduce;
        }

        public Span<double> Softmax(Span<double> vector)
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

        public Span<double> Zeros(params int[] size)
        {
            int length = 1;
            for (int i = 0; i < size.Length; i++) { length *= i; }

            return (double[])Array.CreateInstance(typeof(double), length);
        }

        public Span<double> Ones(params int[] size)
        {
            int length = 1;
            for (int i = 0; i < size.Length; i++) { length *= i; }

            double[] ones = (double[])Array.CreateInstance(typeof(double), length);
            Array.Fill(ones, 1d);

            return ones;
        }
    }
}
