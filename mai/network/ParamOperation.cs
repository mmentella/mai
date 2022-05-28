using mai.blas;

namespace mai.network
{
    public abstract class ParamOperation
        : Operation
    {
        protected Matrix parameter;
        protected Matrix paramGradient;

        public ParamOperation(Matrix parameter)
        {
            this.parameter = parameter;
            paramGradient = default!;
        }

        public override Matrix Backward(Matrix outputGradient)
        {
            AssertSameShape(output, outputGradient);

            inputGradient = InputGradient(outputGradient);
            paramGradient = ParamGradient(outputGradient);

            AssertSameShape(input, inputGradient);
            AssertSameShape(parameter, paramGradient);

            return inputGradient;
        }

        public abstract Matrix ParamGradient(Matrix outputGradient);

        public Matrix GetParamGradient() => paramGradient;
        public Matrix GetParameter() => parameter;
    }

    public class WeightMultiply
        : ParamOperation
    {
        public WeightMultiply(Matrix weights)
            : base(weights) { }

        public override Matrix Output()
        {
            return parameter * input;
        }

        public override Matrix ParamGradient(Matrix outputGradient)
        {
            return input.Transpose() * outputGradient;
        }

        public override Matrix InputGradient(Matrix outputGradient)
        {
            return outputGradient * parameter.Transpose();
        }
    }

    public class BiasAdd
        : ParamOperation
    {
        public BiasAdd(Matrix bias)
            : base(bias)
        {
            if (bias.Rows == 1) { return; }

            throw new InvalidOperationException();
        }

        public override Matrix InputGradient(Matrix outputGradient)
        {
            return Matrix.Ones(input).Hadamard(outputGradient);
        }

        public override Matrix Output()
        {
            return input + parameter;
        }

        public override Matrix ParamGradient(Matrix outputGradient)
        {
            paramGradient = Matrix.Ones(parameter).Hadamard(outputGradient);
            return paramGradient.SumRows();
        }
    }

    public class Sigmoid
        : Operation
    {
        public override Matrix InputGradient(Matrix outputGradient)
        {
            inputGradient = output * (1 - output) * outputGradient;

            return inputGradient;
        }

        public override Matrix Output()
        {
            return input.Sigmoid();
        }
    }

    public class Linear
        : Operation
    {
        public override Matrix InputGradient(Matrix outputGradient)
        {
            return outputGradient;
        }

        public override Matrix Output()
        {
            return input;
        }
    }
}
