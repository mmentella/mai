using mai.blas;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mai.network
{
    public class NeuralNetwork
    {
        private readonly IEnumerable<Layer> layers;
        private readonly Loss loss;
        private readonly int seed;

        public NeuralNetwork(IEnumerable<Layer> layers, Loss loss, int seed = 1)
        {
            this.layers = layers;
            this.loss = loss;
            this.seed = seed;

            foreach(var layer in layers)
            {
                layer.Seed = seed;
            }
        }

        public Matrix Forward(Matrix batch)
        {
            Matrix forward = batch;
            foreach (var layer in layers)
            {
                forward = layer.Forward(forward);
            }

            return forward;
        }

        public void Backward(Matrix lossGradient)
        {
            Matrix gradient = lossGradient;
            foreach (var layer in layers.Reverse())
            {
                gradient = layer.Backward(gradient);
            }
        }

        public double Train(Matrix samples, Matrix labels)
        {
            Matrix prediction = Forward(samples);
            double loss = this.loss.Forward(prediction, labels);

            Backward(this.loss.Backward());

            return loss;
        }

        public IEnumerable<Matrix> Parameters
        {
            get
            {
                var parameters = layers.Select(l=>l.GetParameters());
                return parameters.SelectMany(p => p);
            }
        }

        public IEnumerable<Matrix> ParamGradients
        {
            get
            {
                var parameters = layers.Select(l => l.ParamGradients());
                return parameters.SelectMany(p => p);
            }
        }
    }
}
