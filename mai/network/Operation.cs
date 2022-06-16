using FluentAssertions;
using mai.blas;

namespace mai.network
{
    public abstract class Operation
    {
        protected Matrix input;
        protected Matrix output;
        protected Matrix inputGradient;

        public virtual Matrix Forward(Matrix input)
        {
            this.input = input;
            output = Output();

            return output;
        }

        public virtual Matrix Backward(Matrix outputGradient)
        {
            AssertSameShape(output, outputGradient);

            inputGradient = InputGradient(outputGradient);

            AssertSameShape(input, inputGradient);

            return inputGradient;
        }

        public abstract Matrix InputGradient(Matrix outputGradient);

        public abstract Matrix Output();

        protected virtual void AssertSameShape(Matrix left, Matrix right)
        {
            left.Rows.Should().Be(right.Rows);
            left.Columns.Should().Be(right.Columns);
        }
    }

    public class Dropout
        : Operation
    {
        private Matrix mask;
        private readonly double keep;

        public Dropout(double keep)
        {
            this.keep = keep;
            mask = null!;
        }

        public override Matrix InputGradient(Matrix outputGradient) => outputGradient.Hadamard(mask);

        public override Matrix Output() => input.Hadamard(Mask);

        private Matrix Mask
        {
            get
            {
                Random rand = new();
                mask = new(input.Rows, input.Columns);
                for (int l = 0; l < mask.Length; l++)
                {
                    mask[l] = rand.NextDouble() < keep ? 1 : 0;
                }

                return mask;
            }
        }
    }
}
