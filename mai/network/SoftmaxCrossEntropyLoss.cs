using mai.blas;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mai.network
{
    public class SoftmaxCrossEntropyLoss
        : Loss
    {
        private readonly double epsilon;
        private readonly bool singleOutput;
        private Matrix softmaxPrediction;

        public SoftmaxCrossEntropyLoss(double epsilon = 1e-9)
        {
            this.epsilon = epsilon;
            singleOutput = false;
        }

        public override Matrix InputGradient() => softmaxPrediction - target;

        public override double Output()
        {
            softmaxPrediction = prediction.Softmax(epsilon, 1 - epsilon);
            var softmaxCrossEntropy = softmaxPrediction.Log().Hadamard(-1d * target) - (1 - target).Hadamard((1 - softmaxPrediction).Log());

            return softmaxCrossEntropy.Sum();
        }
    }
}
