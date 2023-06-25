using System.Text;

namespace mai.v1.blas;

public static class PrintExtensions
{
    public static string Print(this Matrix values, int rows, int columns)
    {
        int Offset(int[] index) { return index[0] * columns + index[1]; }
        bool IncrementIndex(int[] index, int? skipAxis = null)
        {
            int[] shape = { rows, columns };
            for (int i = 1; i >= 0; i--)
            {
                if (i == skipAxis) { continue; }

                index[i]++;
                if (index[i] >= shape[i])
                {
                    index[i] = 0;
                    continue;
                }
                return true;
            }
            return false;
        }

        int rank = 2;

        StringBuilder stringBuilder = new();

        //double max = values.Max();
        int numbers = 6;//((int)max).ToString().Length;
        string format = stringBuilder.Append('0', numbers)
                                     .Append(".############")
                                     .ToString();
        int maxLen = 10;
        stringBuilder.Clear();
        string placeholder = stringBuilder.Append(' ', maxLen)
                                          .ToString();
        stringBuilder.Clear();

        stringBuilder.Append('(')
                     .Append('[', rank);

        int[] index = new int[rank];
        string value = placeholder + (values[Offset(index)] == 0 ? "0" :
                                      values[Offset(index)].ToString(format));
        value = value[^maxLen..];
        stringBuilder.Append(value);

        index[rank - 1] = 1;
        do
        {
            int closeParentheses = 0;

            if (index[rank - 1] == 0)
            {
                stringBuilder.Append(']');
                closeParentheses++;

                for (int r = rank - 2; r >= 0; r--)
                {
                    if (index[r] == 0)
                    {
                        stringBuilder.Append(']');
                        closeParentheses++;
                    }
                    else
                    {
                        for (int c = 0; c < closeParentheses; c++)
                        {
                            stringBuilder.Append(Environment.NewLine);
                        }
                        stringBuilder.Append(' ', r + 2)
                                     .Append('[', rank - 1 - r);
                        break;
                    }
                }
            }
            else { stringBuilder.Append(' '); }

            value = placeholder + (values[Offset(index)] == 0 ? "0" :
                                   values[Offset(index)].ToString(format));
            value = value[^maxLen..];
            stringBuilder.Append(value);

        } while (IncrementIndex(index));

        stringBuilder.Append(']', rank)
                     .Append($", Rank {rank}, " +
                             $"Shape [{rows},{columns}]), " +
                             $"Length {rows * columns})");

        return stringBuilder.ToString();
    }
}
