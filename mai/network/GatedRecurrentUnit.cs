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

        private double[] dx;
        private double[] dh;

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

            o = Softmax(h).ToArray();

            return o;
        }

        public double[] Train(double[] inputs,
                              double[] labels,
                              double learningRate = 1.0e-3,
                              int epochs = 100)
        {
            double[] loss = (double[])Array.CreateInstance(typeof(double), epochs);
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double[] output = Forward(inputs);
                BackPropagate(inputs, output, labels);
                UpdateWeights(learningRate);

                loss[epoch] = CalculateLoss(output, labels);
            }

            return loss;
        }

        private double CalculateLoss(double[] output, double[] labels)
        {
            return 0.5 * labels.Zip(output, (l, o) => (l - o) * (l - o)).Sum();
        }

        private void BackPropagate(double[] input, double[] output, double[] outputSamples)
        {
            
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

        public Span<double> DotProduct(Span<double> matrix, Span<double> vector, int rows)
        {
            double[] dot = (double[])Array.CreateInstance(typeof(double), rows);
            for (int i = 0; i < rows; i++)
            {
                dot[i] = Reduce(matrix.Slice(i * vector.Length, vector.Length), vector);
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
