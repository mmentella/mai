using mai.blas;

namespace mai.network
{
    public class SoftmaxCrossEntropyLoss
        : Loss
    {
        private readonly double epsilon;
        private bool singleOutput;
        private Matrix softmaxPrediction;

        public SoftmaxCrossEntropyLoss(double epsilon = 0.000000001f)
        {
            this.epsilon = epsilon;
            singleOutput = false;

            softmaxPrediction = default!;
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

            Matrix softmaxCrossEntropy = softmaxPrediction.Log().Hadamard(-1f * target) -
                                         (1f - softmaxPrediction).Log().Hadamard(1f - target);

            return softmaxCrossEntropy.Sum() / prediction.Rows;
        }
    }
}
