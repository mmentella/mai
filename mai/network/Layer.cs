﻿using mai.blas;

namespace mai.network
{
    public abstract class Layer
    {
        protected Matrix output;
        protected Matrix input;

        protected readonly int neurons;
        protected readonly List<Matrix> parameters;
        protected readonly List<Matrix> parameterGrads;
        protected readonly List<Operation> operations;

        public int? Seed;

        public Layer(int neurons)
        {
            this.neurons = neurons;

            First = true;
            parameters = new List<Matrix>();
            parameterGrads = new List<Matrix>();
            operations = new List<Operation>();

            input = default!;
            output = default!;
        }

        public bool First;

        protected abstract void SetupLayer(Matrix input);

        public virtual Matrix Forward(Matrix input)
        {
            if (First)
            {
                SetupLayer(input);
                First = false;
            }
            this.input = input;

            foreach (var operation in operations)
            {
                input = operation.Forward(input);
            }

            output = input;
            return output;
        }

        public virtual Matrix Backward(Matrix outputGradient)
        {
            Matrix.SameShape(output, outputGradient);

            foreach (var operation in operations.AsEnumerable().Reverse())
            {
                outputGradient = operation.Backward(outputGradient);
            }

            ParamGradients();

            return outputGradient;
        }

        public virtual IEnumerable<Matrix> ParamGradients()
        {
            parameterGrads.Clear();
            parameterGrads.AddRange(
                operations.Where(o => o is ParamOperation)
                          .Cast<ParamOperation>()
                          .Select(p => p.GetParamGradient())
            );

            return parameterGrads.AsEnumerable();
        }

        public IEnumerable<Matrix> GetParameters()
        {
            parameters.Clear();
            parameters.AddRange(
                operations.Where(o => o is ParamOperation)
                          .Cast<ParamOperation>()
                          .Select(p => p.GetParameter())
            );

            return parameters.AsEnumerable();
        }
    }

    public class Dense
        : Layer
    {
        private readonly double dropout;
        private readonly Operation activation;

        public Dense(int neurons, Operation activation = null!, int? seed = null!, double dropout = 1)
            : base(neurons)
        {
            this.dropout = dropout;
            this.activation = activation ?? new Sigmoid();
            Seed = seed;
        }

        protected override void SetupLayer(Matrix input)
        {
            parameters.Clear();

            Matrix weights = new(input.Columns, neurons);
            weights.InitRandom(seed: Seed, stdDev: 2d / (weights.Rows + weights.Columns));

            Matrix bias = new(1, neurons);
            bias.InitRandom(seed: Seed, stdDev: 2d / (weights.Rows + weights.Columns));

            parameters.Add(weights);
            parameters.Add(bias);

            operations.Add(new WeightMultiply(weights));
            operations.Add(new BiasAdd(bias));
            operations.Add(activation);

            if (dropout < 1)
            {
                operations.Add(new Dropout(dropout));
            }
        }
    }
}
