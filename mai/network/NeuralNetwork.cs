using mai.blas;

namespace mai.network
{
    public class NeuralNetwork
    {
        private readonly int seed;

        public NeuralNetwork(IEnumerable<Layer> layers, Loss loss, int seed = 1)
        {
            Layers = layers;
            Loss = loss;
            this.seed = seed;

            foreach (var layer in layers)
            {
                layer.Seed = seed;
            }
        }

        public IEnumerable<Layer> Layers { get; }
        public Loss Loss { get; }

        public Matrix Forward(Matrix batch)
        {
            Matrix forward = batch;
            foreach (var layer in Layers)
            {
                forward = layer.Forward(forward);
            }

            return forward;
        }

        public void Backward(Matrix lossGradient)
        {
            Matrix gradient = lossGradient;
            foreach (var layer in Layers.Reverse())
            {
                gradient = layer.Backward(gradient);
            }
        }

        public double Train(Matrix samples, Matrix labels)
        {
            Matrix prediction = Forward(samples);
            double loss = Loss.Forward(prediction, labels);

            Backward(Loss.Backward());

            return loss;
        }

        public IEnumerable<Matrix> Parameters
        {
            get
            {
                var parameters = Layers.Select(l => l.GetParameters());
                return parameters.SelectMany(p => p);
            }
        }

        public IEnumerable<Matrix> ParamGradients
        {
            get
            {
                var parameters = Layers.Select(l => l.ParamGradients());
                return parameters.SelectMany(p => p);
            }
        }
    }
}
