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

        public virtual void Update(int percentage) { }
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
                for (int l = 0; l < parameter.Length; l++)
                {
                    parameter[l] -= learningRate * gradient[l];
                }
            }
        }
    }

    public class SGDMomentum
        : SGD
    {
        public SGDMomentum(NeuralNetwork network, double learningRate)
            : base(network, learningRate) { }

        public override void Step()
        {
            base.Step();
        }

        override public void Update(int percentage) { learningRate = learningRate * (1 - 0.01 * percentage); }
    }
}
