using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace maiTests.memory
{
    public class MarshallTests
    {
        [Fact]
        public void MarshallTest()
        {
            IntPtr data = Marshal.AllocHGlobal(sizeof(float) * 10);
            unsafe
            {
                ((float*)data.ToPointer())[0] = 0;
                ((float*)data.ToPointer())[1] = 1;
                ((float*)data.ToPointer())[2] = 2;
                ((float*)data.ToPointer())[3] = 3;
                ((float*)data.ToPointer())[4] = 4;
                ((float*)data.ToPointer())[5] = 5;
                ((float*)data.ToPointer())[6] = 6;
                ((float*)data.ToPointer())[7] = 7;
                ((float*)data.ToPointer())[8] = 8;
                ((float*)data.ToPointer())[9] = 9;
            }

            unsafe
            {
                Debug.WriteLine(((float*)data.ToPointer())[0]);
                Debug.WriteLine(((float*)data.ToPointer())[1]);
                Debug.WriteLine(((float*)data.ToPointer())[2]);
                Debug.WriteLine(((float*)data.ToPointer())[3]);
                Debug.WriteLine(((float*)data.ToPointer())[4]);
                Debug.WriteLine(((float*)data.ToPointer())[5]);
                Debug.WriteLine(((float*)data.ToPointer())[6]);
                Debug.WriteLine(((float*)data.ToPointer())[7]);
                Debug.WriteLine(((float*)data.ToPointer())[8]);
                Debug.WriteLine(((float*)data.ToPointer())[9]);
            }

            Marshal.FreeHGlobal(data);
        }
    }
}
