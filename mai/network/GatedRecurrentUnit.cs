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

        private double[] htilde;

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
            htilde = new double[stateLength];

            RandomInitialization();
        }

        public double[] Forward(double[] input)
        {
            r = Sigmoid(Sum(DotProduct(wr, h), DotProduct(ur, input))).ToArray();
            z = Sigmoid(Sum(DotProduct(wz, h), DotProduct(uz, input))).ToArray();

            htilde = HyperbolicTanget(Sum(DotProduct(wh, Hadamard(r, h)), DotProduct(uh, input))).ToArray();

            h = Sum(Hadamard(Less(1, z), htilde), Hadamard(z, h)).ToArray();

            return Softmax(h).ToArray();
        }

        private void RandomInitialization()
        {
            Random rnd = new();

            ur.AsSpan().Fill(1.0e-6 * rnd.NextDouble());
            uz.AsSpan().Fill(1.0e-6 * rnd.NextDouble());
            uh.AsSpan().Fill(1.0e-6 * rnd.NextDouble());

            wr.AsSpan().Fill(1.0e-6 * rnd.NextDouble());
            wz.AsSpan().Fill(1.0e-6 * rnd.NextDouble());
            wh.AsSpan().Fill(1.0e-6 * rnd.NextDouble());

            h.AsSpan().Fill(1.0e-6 * rnd.NextDouble());
            htilde.AsSpan().Fill(1.0e-6 * rnd.NextDouble());
        }

        private Span<double> Sigmoid(Span<double> values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = 1 / (1 + Math.Exp(-values[i]));
            }

            return values;
        }

        private Span<double> HyperbolicTanget(Span<double> values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (Math.Exp(values[i]) - Math.Exp(-values[i])) /
                            (Math.Exp(values[i]) + Math.Exp(-values[i]));
            }

            return values;
        }

        private Span<double> Sum(Span<double> left, Span<double> right)
        {
            double[] sum = (double[])Array.CreateInstance(typeof(double), left.Length);
            for (int i = 0; i < left.Length; i++)
            {
                sum[i] = left[i] + right[i];
            }

            return sum;
        }

        private Span<double> Less(double left, Span<double> right)
        {
            double[] less = (double[])Array.CreateInstance(typeof(double), right.Length);
            for (int i = 0; i < right.Length; i++)
            {
                less[i] = left - right[i];
            }

            return less;
        }

        private Span<double> Hadamard(Span<double> left, Span<double> right)
        {
            double[] hadamard = (double[])Array.CreateInstance(typeof(double), left.Length);
            for (int i = 0; i < left.Length; i++)
            {
                hadamard[i] = left[i] * right[i];
            }

            return hadamard;
        }

        private Span<double> DotProduct(Span<double> matrix, Span<double> vector)
        {
            double[] dot = (double[])Array.CreateInstance(typeof(double), vector.Length);
            for (int i = 0; i < dot.Length; i++)
            {
                dot[i] = Reduce(matrix.Slice(i * vector.Length, vector.Length), vector);
            }

            return dot;
        }

        private double Reduce(Span<double> left, Span<double> right)
        {
            double reduce = 0;
            for (int i = 0; i < left.Length; i++)
            {
                reduce += left[i] * right[i];
            }

            return reduce;
        }

        private Span<double> Softmax(Span<double> vector)
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
    }
}
