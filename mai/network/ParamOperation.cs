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
            return input * parameter;
        }

        public override Matrix ParamGradient(Matrix outputGradient)
        {
            Matrix transpose = input.Transpose();
            return transpose * outputGradient;
        }

        public override Matrix InputGradient(Matrix outputGradient)
        {
            Matrix transpose = parameter.Transpose();
            return outputGradient * transpose;
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
            return outputGradient;
        }

        public override Matrix Output()
        {
            return input + parameter;
        }

        public override Matrix ParamGradient(Matrix outputGradient)
        {
            paramGradient = outputGradient;

            Matrix result = paramGradient.SumRows();
            result.Reshape(1, parameter.Columns);

            return result;
        }
    }

    public class Sigmoid
        : Operation
    {
        public override Matrix InputGradient(Matrix outputGradient)
        {
            var sigmoidGradient = output.Hadamard(1 - output);
            inputGradient = sigmoidGradient.Hadamard(outputGradient);

            return inputGradient;
        }

        public override Matrix Output()
        {
            return input.Sigmoid();
        }
    }

    public class Tanh
        : Operation
    {
        public override Matrix InputGradient(Matrix outputGradient)
        {
            var tanhGradient = 1 - output.Square();
            inputGradient = tanhGradient.Hadamard(outputGradient);

            return inputGradient;
        }

        public override Matrix Output()
        {
            return input.Tanh();
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
