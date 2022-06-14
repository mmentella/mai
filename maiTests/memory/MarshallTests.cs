using mai.blas;
using System;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Xunit;

namespace maiTests.memory
{
    public class MarshallTests
    {
        [Fact]
        public void DotTest()
        {
            Matrix left = Enumerable.Range(1, 5).Select(i => (double)i).ToArray();
            left.Reshape(1, 5);

            Matrix right = Enumerable.Range(1, 15).Select(i => (double)i).ToArray();
            right = right.Reshape(5, 3);

            Matrix dot = new(left.Rows, right.Columns);

            right = right.Transpose();

            Span<double> lspan = left;
            Span<double> rspan = right;
            for (int r = 0; r < dot.Rows; r++)
            {
                Span<Vector<double>> lsv = MemoryMarshal.Cast<double, Vector<double>>(lspan.Slice(r * left.Columns, left.Columns));
                for (int c = 0; c < dot.Columns; c++)
                {
                    Span<Vector<double>> rsv = MemoryMarshal.Cast<double, Vector<double>>(rspan.Slice(c * right.Columns, right.Columns));
                    for (int s = 0; s < lsv.Length; s++)
                    {
                        dot[r, c] = Vector.Dot(lsv[s], rsv[s]);
                    }

                    for (int s = lsv.Length * Vector<double>.Count; s < left.Columns; s++)
                    {
                        dot[r, c] += left[r, s] * right[c, s];
                    }
                }
            }

        }

        [Fact]
        public void TrasnposeTest()
        {
            Matrix m = Enumerable.Range(0, 1200).Select(i => (double)i).ToArray();
            m = m.Reshape(40, 30);

            Matrix t = m.Transpose();
        }

        [Fact]
        public void MarshallTest()
        {
            IntPtr data = Marshal.AllocHGlobal(sizeof(double) * 10);
            unsafe
            {
                ((double*)data.ToPointer())[0] = 0;
                ((double*)data.ToPointer())[1] = 1;
                ((double*)data.ToPointer())[2] = 2;
                ((double*)data.ToPointer())[3] = 3;
                ((double*)data.ToPointer())[4] = 4;
                ((double*)data.ToPointer())[5] = 5;
                ((double*)data.ToPointer())[6] = 6;
                ((double*)data.ToPointer())[7] = 7;
                ((double*)data.ToPointer())[8] = 8;
                ((double*)data.ToPointer())[9] = 9;
            }

            unsafe
            {
                Debug.WriteLine(((double*)data.ToPointer())[0]);
                Debug.WriteLine(((double*)data.ToPointer())[1]);
                Debug.WriteLine(((double*)data.ToPointer())[2]);
                Debug.WriteLine(((double*)data.ToPointer())[3]);
                Debug.WriteLine(((double*)data.ToPointer())[4]);
                Debug.WriteLine(((double*)data.ToPointer())[5]);
                Debug.WriteLine(((double*)data.ToPointer())[6]);
                Debug.WriteLine(((double*)data.ToPointer())[7]);
                Debug.WriteLine(((double*)data.ToPointer())[8]);
                Debug.WriteLine(((double*)data.ToPointer())[9]);
            }

            Marshal.FreeHGlobal(data);
        }
    }
}
