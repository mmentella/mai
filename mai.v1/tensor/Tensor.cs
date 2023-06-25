//using System.Buffers;
//using System.Runtime.CompilerServices;
//using System.Text;

//namespace mai.v1.Matrix;

//public class Matrix
//{
//    private readonly double[] data;

//    public Matrix(params int[] shape)
//        : this(shape, null, null) { }

//    public Matrix(double[] data, int[] shape)
//        : this(shape, data, null) { }

//    public Matrix(int[] shape, double[]? data = default, int[]? stride = default)
//    {
//        Shape = shape;
//        Length = Shape.Length == 0 ? 0 :
//                 Shape.Aggregate((i, j) => i * j);
//        Stride = stride ?? BuildStride();

//        this.data = data ?? new double[Length];
//    }

//    public int[] Shape { get; }
//    public int[] Stride { get; }
//    public int Length { get; }
//    public int Rank => Shape.Length;

//    public double this[params int[] indices]
//    {
//        get
//        {
//            indices.LessThanOrEqualsCheck(Shape);

//            int offset = Offset(indices);
//            return data[offset];
//        }
//        set
//        {
//            indices.LessThanOrEqualsCheck(Shape);

//            int offset = Offset(indices);
//            data[offset] = value;
//        }
//    }

//    public double this[int index]
//    {
//        get => data[index];
//        set => data[index] = value;
//    }

//    public Matrix Slice(int[] begin, int[] size)
//    {
//        int[] lastIndices = begin.Zip(size, (b, s) => b + s)
//                                 .ToArray();
//        lastIndices.LessThanOrEqualsCheck(Shape);

//        int start = Offset(begin);
//        int last = Offset(lastIndices);

//        Matrix Matrix = data.Skip(start)
//                            .Take(last)
//                            .ToArray()
//                            .AsMatrix(size);
//        return Matrix;
//    }

//    public Queue<double[]> BufferData(int axis)
//    {
//        (0 <= axis && axis <= Rank).IfNotThrow<InvalidOperationException>();

//        int[] current = new int[Rank];
//        Queue<double[]> axes = new();

//        do
//        {
//            double[] data = new double[Shape[axis]];
//            for (int i = 0; i < Shape[axis]; i++)
//            {
//                current[axis] = i;
//                data[i] = this.data[Offset(current)];
//            }

//            axes.Enqueue(data);
//        } while (IncrementIndex(current, axis));

//        return axes;
//    }

//    public Matrix Reshape(params int[] shape)
//    {
//        Matrix Matrix = data.ToArray()
//                            .AsMatrix(shape);
//        return Matrix;
//    }

//    public Matrix Reduce()
//    {
//        int[] shape = Shape.Skip(1).ToArray();
//        int[] stride = Stride.Skip(1).ToArray();

//        Matrix Matrix = data.ToArray()
//                            .AsMatrix(shape, stride);
//        return Matrix;
//    }


//    public Matrix Contraction(Matrix right, params int[][] axes)
//    {
//        (axes[0].Length == axes[1].Length)
//            .IfNotThrow<InvalidOperationException>();
//        axes[0].Zip(axes[1], (l, r) => Shape[l] == right.Shape[r])
//               .All(b => b)
//               .IfNotThrow<InvalidOperationException>();

//        List<int> shapeListLeft = Shape.ToList();
//        Array.ForEach(axes[0], a => shapeListLeft.RemoveAt(a));

//        List<int> shapeListRight = right.Shape.ToList();
//        Array.ForEach(axes[1], a => shapeListRight.RemoveAt(a));

//        int[] shape = shapeListLeft.Concat(shapeListRight)
//                                   .ToArray();

//        int[] axesLeft = Enumerable.Range(0, Rank)
//                                   .Where(r => !axes[0].Contains(r))
//                                   .Concat(axes[0])
//                                   .ToArray();
//        int dimLeft = 1;
//        Array.ForEach(axes[0], a => dimLeft *= Shape[a]);
//        int[] shapeLeft = { shapeListLeft.Aggregate((i, j) => i * j), dimLeft };

//        int[] axesRight = axes[1].Concat(Enumerable.Range(0, right.Rank)
//                                                   .Where(r => !axes[1].Contains(r)))
//                                 .ToArray();
//        int dimRight = 1;
//        Array.ForEach(axes[1], a => dimRight *= right.Shape[a]);
//        int[] shapeRight = { dimRight, shapeListRight.Aggregate((i, j) => i * j) };

//        Matrix a = Transpose(axesLeft).Reshape(shapeLeft);
//        Matrix b = right.Transpose(axesRight).Reshape(shapeRight);

//        Matrix Matrix = a.Dot(b);

//        Matrix = Matrix.Reshape(shape);
//        return Matrix;
//    }

//    public Matrix Dot(Matrix right)
//    {
//        (Rank == right.Rank && Rank == 2).IfNotThrow<InvalidOperationException>();

//        Queue<double[]> rows = BufferData(1);
//        Queue<double[]> cols = right.BufferData(0);

//        int idx = 0;
//        double[] data = new double[Shape[0] * right.Shape[1]];
//        foreach (var row in rows)
//        {
//            foreach (var col in cols)
//            {
//                data[idx] += row.Zip(col, (r, c) => r * c).Sum();
//                idx++;
//            }
//        }

//        int[] shape = { Shape[0], right.Shape[1] };

//        return data.AsMatrix(shape);
//    }

//    public Matrix Transpose(params int[] axes)
//    {
//        if (Rank == 0) { return this; }

//        if (axes == null || axes.Length == 0)
//        {
//            axes = Enumerable.Range(0, Rank)
//                             .Reverse()
//                             .ToArray();
//        }

//        int[] shape = new int[Rank];
//        int[] stride = new int[Rank];
//        Array.ForEach(axes, a =>
//        {
//            shape[a] = Shape[axes[a]];
//            stride[a] = Stride[axes[a]];
//        });

//        Matrix Matrix = new(shape, stride: stride);
//        Matrix.Load(this);

//        return Matrix;
//    }

//    public void Fill(double value) =>
//        Array.Fill(data, value);
//    public void FillWithRange()
//    {
//        for (int l = 0; l < Length; l++) { data[l] = l; }
//    }

//    public void Load(Matrix Matrix)
//    {
//        int[] index = new int[Matrix.Rank];
//        int current = 0;
//        do
//        {
//            this[index] = Matrix[Offset(index)/*current*/];
//            current++;
//        } while (IncrementIndex(index));
//    }

//    public string Print()
//    {
//        StringBuilder stringBuilder = new();

//        double max = data.Max();
//        int numbers = ((int)max).ToString().Length;
//        string format = stringBuilder.Append('0', numbers)
//                                     .Append(".############")
//                                     .ToString();
//        int maxLen = max.ToString(format)
//                        .TrimStart('0')
//                        .Length;
//        stringBuilder.Clear();
//        string placeholder = stringBuilder.Append(' ', maxLen)
//                                          .ToString();
//        stringBuilder.Clear();

//        stringBuilder.Append('(')
//                     .Append('[', Rank);

//        int[] index = new int[Rank];
//        string value = placeholder + (this[index] == 0 ? "0" :
//                                      this[index].ToString(format)
//                                                 .TrimStart('0'));
//        value = value[^maxLen..];
//        stringBuilder.Append(value);

//        index[Rank - 1] = 1;
//        do
//        {
//            int closeParentheses = 0;

//            if (index[Rank - 1] == 0)
//            {
//                stringBuilder.Append(']');
//                closeParentheses++;

//                for (int r = Rank - 2; r >= 0; r--)
//                {
//                    if (index[r] == 0)
//                    {
//                        stringBuilder.Append(']');
//                        closeParentheses++;
//                    }
//                    else
//                    {
//                        for (int c = 0; c < closeParentheses; c++)
//                        {
//                            stringBuilder.Append(Environment.NewLine);
//                        }
//                        stringBuilder.Append(' ', r + 2)
//                                     .Append('[', Rank - 1 - r);
//                        break;
//                    }
//                }
//            }
//            else { stringBuilder.Append(' '); }

//            value = placeholder + (this[index] == 0 ? "0" :
//                                   this[index].ToString(format)
//                                              .TrimStart('0'));
//            value = value[^maxLen..];
//            stringBuilder.Append(value);

//        } while (IncrementIndex(index));

//        stringBuilder.Append(']', Rank)
//                     .Append($", Rank {Rank}, " +
//                             $"Shape [{string.Join(", ", Shape)}]), " +
//                             $"Length [{string.Join(", ", Length)}])");

//        return stringBuilder.ToString();
//    }

//    private int[] BuildStride()
//    {
//        int currentStride = 1;
//        int[] stride = new int[Shape.Length];
//        for (int i = stride.Length - 1; i >= 0; i--)
//        {
//            stride[i] = currentStride;
//            currentStride *= Shape[i];
//        }

//        return stride;
//    }

//    [MethodImpl(MethodImplOptions.AggressiveInlining)]
//    private int Offset(params int[] indices) =>
//        indices.Zip(Stride, (i, s) => i * s)
//               .Sum();

//    [MethodImpl(MethodImplOptions.AggressiveInlining)]
//    private bool IncrementIndex(int[] index, int? skipAxis = null)
//    {
//        for (int i = Rank - 1; i >= 0; i--)
//        {
//            if (i == skipAxis) { continue; }

//            index[i]++;
//            if (index[i] >= Shape[i])
//            {
//                index[i] = 0;
//                continue;
//            }
//            return true;
//        }
//        return false;
//    }
//}
