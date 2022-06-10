namespace mai.network
{
    public abstract class Optimizer
    {
        protected float learningRate;

        public Optimizer(NeuralNetwork network, float learningRate)
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
        public SGD(NeuralNetwork network, float learningRate = 0.01f)
            : base(network, learningRate)
        {
        }

        public override void Step()
        {
            var pg = Network.Parameters.Zip(Network.ParamGradients, (parameter, gradient) => (parameter, gradient));
            foreach (var (parameter, gradient) in pg)
            {
                for (int l = 0; l < parameter.Length; l++)
                {
                    parameter[l] -= learningRate * gradient[l];
                }
            }
        }
    }
}
