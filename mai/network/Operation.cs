﻿using mai.blas;

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
            if (left.Rows == right.Rows && left.Columns == right.Columns) { return; }

            throw new InvalidOperationException();
        }
    }
}
