using mai.blas;

namespace mai.network
{
    public abstract class Loss
    {
        protected Matrix prediction;
        protected Matrix target;

        protected Matrix inputGradient;

        public double Forward(Matrix prediction, Matrix target)
        {
            Matrix.SameShape(prediction, target);

            this.prediction = prediction;
            this.target = target;

            double loss = Output();

            return loss;
        }

        public Matrix Backward()
        {
            inputGradient = InputGradient();
            Matrix.SameShape(inputGradient, prediction);

            return inputGradient;
        }

        public abstract double Output();
        public abstract Matrix InputGradient();
    }

    public class MeanSquaredError
        : Loss
    {
        public override Matrix InputGradient()
        {
            return 2f * ((prediction - target) / prediction.Rows);
        }

        public override double Output()
        {
            var loss = (prediction - target).Square().Sum() / prediction.Rows;
            return loss;
        }
    }
}
