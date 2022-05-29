using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mai.network
{
    public abstract class Optimizer
    {
        protected NeuralNetwork network;
        protected double learningRate;

        public Optimizer(NeuralNetwork network, double learningRate)
        {
            this.network = network;
            this.learningRate = learningRate;
        }

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
            var pg = network.Parameters.Zip(network.ParamGradients, (parameter, gradient) => (parameter, gradient));
            foreach (var (parameter, gradient) in pg)
            {
                parameter.Run(parameter, learningRate * gradient, (p, g, r, c) => p[r, c] -= g[r, c]);
            }
        }
    }
}
