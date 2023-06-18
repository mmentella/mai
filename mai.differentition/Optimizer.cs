using System.Diagnostics;

namespace mai.differentiation;

public class Optimizer
{
    private readonly Neuron neuron;

    public Optimizer()
    {
        neuron = new();
    }

    public Neuron Train(double input, double output, int epochs, double learningRate)
    {
        Debug.IndentLevel = 1;
        for (int i = 0; i < epochs; i++)
        {
            neuron.Forward(input);
            double prediction = neuron.Output;
            double loss = MeanSquaredError(output, prediction);

            if (loss < 0.0001)
            {
                break;
            }

            double outputGradient = MeanSquaredErrorGradient(output, prediction);

            neuron.Backward(outputGradient, learningRate);

            Debug.WriteLine($"epoch: {i:0}|output: {output:0}|prediction: {prediction:0.0000}|loss: {loss:0.0000}|neuron: {neuron}");
        }

        return neuron;
    }

    private double MeanSquaredErrorGradient(double output, double prediction)
    => prediction - output;

    private static double MeanSquaredError(double output, double prediction)
        => 0.5 * Math.Pow(output - prediction, 2);
}
