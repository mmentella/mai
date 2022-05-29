namespace mai.network
{
    public abstract class Optimizer
    {
        protected double learningRate;

        public Optimizer(NeuralNetwork network, double learningRate)
        {
            Network = network;
            this.learningRate = learningRate;
        }
        public NeuralNetwork Network { get; }

        public abstract void Step();
    }

    public class SGD
        : Optimizer
    {
        public SGD(NeuralNetwork network, double learningRate = 0.01)
            : base(network, learningRate)
        {
        }

        public override void Step()
        {
            var pg = Network.Parameters.Zip(Network.ParamGradients, (parameter, gradient) => (parameter, gradient));
            foreach (var (parameter, gradient) in pg)
            {
                parameter.Run(parameter, gradient, (p, g, r, c) => p[r, c] -= learningRate * g[r, c]);
            }
        }
    }
}
