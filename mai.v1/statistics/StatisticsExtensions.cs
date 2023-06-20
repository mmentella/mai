namespace mai.v1.statistics;

public static class StatisticsExtensions
{
    public static double StandardDeviation(this IEnumerable<double> values)
    {
        double avg = values.Average();
        double sum = values.Sum(d => Math.Pow(d - avg, 2));
        return Math.Sqrt(sum / values.Count());
    }
}
