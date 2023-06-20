using mai.v1.statistics;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace maiTests.v1;

public class DataTest
{
    [Fact]
    public async Task LoadTest()
    {
        List<double> prices = new();
        string[] lines = await File.ReadAllLinesAsync("v1\\EUR.USD-Minute-Trade.txt");
        lines.Select(line => line.Split(','))
            .ToList()
            .ForEach(line => 
            {
                double open = double.Parse(line[2], CultureInfo.InvariantCulture);
                double high = double.Parse(line[3], CultureInfo.InvariantCulture);
                double low = double.Parse(line[4], CultureInfo.InvariantCulture);
                double close = double.Parse(line[5], CultureInfo.InvariantCulture);

                prices.Add(open);
                prices.Add(high);
                prices.Add(low);
                prices.Add(close);
            });
        double min = prices.Min();
        double max = prices.Max();
        double avg = prices.Average();
        double std = prices.StandardDeviation();

        double[] distinct = prices.Distinct().ToArray();
        double[][] hotencoded = HotEncode(distinct);
    }

    private double[][] HotEncode(double[] distinct)
    {
        double[][] hotencoded = new double[distinct.Length][];
        for (int i = 0; i < distinct.Length; i++)
        {
            hotencoded[i] = new double[distinct.Length];
            hotencoded[i][i] = 1;
        }
        return hotencoded;
    }
}
