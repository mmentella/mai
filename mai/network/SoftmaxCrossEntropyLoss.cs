using mai.blas;

namespace mai.network
{
    public class SoftmaxCrossEntropyLoss
        : Loss
    {
        private readonly double epsilon;
        private bool singleOutput;
        private Matrix softmaxPrediction;

        public SoftmaxCrossEntropyLoss(double epsilon = 0.000000001)
        {
            this.epsilon = epsilon;
            singleOutput = false;

            softmaxPrediction = default;
        }

        public override Matrix InputGradient()
        {
            if (singleOutput)
            {
                return softmaxPrediction - target;
            }
            else
            {
                return (softmaxPrediction - target) / prediction.Rows;
            }
        }

        public override double Output()
        {
            if (target.Columns == 1) { singleOutput = true; }

            if (singleOutput)
            {
                prediction = prediction.Normalize();
                target = target.Normalize();
            }

            softmaxPrediction = prediction.Softmax(epsilon, 1 - epsilon);

            Matrix softmaxCrossEntropy = softmaxPrediction.Log().Hadamard(-1d * target) -
                                         (1d - softmaxPrediction).Log().Hadamard(1d - target);

            return softmaxCrossEntropy.Sum() / prediction.Rows;
        }
    }
}
