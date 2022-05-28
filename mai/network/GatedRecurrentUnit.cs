using mai.blas;

namespace mai.network
{
    public class GatedRecurrentUnit
    {
        private Matrix wz;
        private Matrix wr;
        private Matrix wh;

        private Matrix uz;
        private Matrix ur;
        private Matrix uh;

        private Matrix bz;
        private Matrix br;
        private Matrix bh;

        private Matrix z;
        private Matrix r;
        private Matrix h;
        private Matrix s;

        private Matrix o;
        private Matrix v;
        private Matrix bv;

        private readonly int inputLength;
        private readonly int memoryLength;

        public GatedRecurrentUnit(int inputLength, int memoryLength)
        {
            uz = new Matrix(memoryLength, inputLength);
            ur = new Matrix(memoryLength, inputLength);
            uh = new Matrix(memoryLength, inputLength);

            wz = new Matrix(memoryLength, memoryLength);
            wr = new Matrix(memoryLength, memoryLength);
            wh = new Matrix(memoryLength, memoryLength);

            bz = new Matrix(memoryLength, 1);
            br = new Matrix(memoryLength, 1);
            bh = new Matrix(memoryLength, 1);

            z = default!;
            r = default!;
            h = default!;
            o = default!;

            s = new Matrix(memoryLength, 1);

            v = new Matrix(inputLength, memoryLength);
            bv = new Matrix(inputLength, 1);

            this.inputLength = inputLength;
            this.memoryLength = memoryLength;

            RandomInitialization();
        }

        public Matrix Forward(Matrix input)
        {
            z = (uz * input + wz * s + bz).Sigmoid();
            r = (ur * input + wr * s + br).Sigmoid();
            h = (uh * input + wh * s.Hadamard(r) + bh).Tanh();
            s = (1 - z).Hadamard(h) + z.Hadamard(s);

            o = v * s + bv;

            return o;
        }

        public double[] Train(IList<(Matrix sample, Matrix label)> trainingSet,
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

        private void BackPropagate(IList<(Matrix sample, Matrix label)> trainingSet, double learningRate)
        {
            int timestep = 0;
            Dictionary<int, (Matrix z, Matrix r, Matrix h, Matrix s, Matrix o, Matrix label, Matrix sample)> memory = new();
            foreach (var (sample, label) in trainingSet)
            {
                Forward(sample);

                timestep++;
                memory.Add(timestep, (z, r, h, s, o, label, sample));
            }

            for (int t = timestep; t > 1; t--)
            {

            }
        }

        private void RandomInitialization()
        {
            uz.InitRandom();
            ur.InitRandom();
            uh.InitRandom();

            wz.InitRandom();
            wr.InitRandom();
            wh.InitRandom();

            bz.InitRandom();
            br.InitRandom();
            bh.InitRandom();

            v.InitRandom();
            bv.InitRandom();
        }


    }
}
