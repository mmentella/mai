using Xunit;
using mai.network;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mai.network.Tests
{
    public class GatedRecurrentUnitTests
    {
        [Fact()]
        public void ForwardTest()
        {
            double[] input = { 1, 2, 3, 4, 5 };
            GatedRecurrentUnit gru = new GatedRecurrentUnit(5, 5);

            double[] output = gru.Forward(input);
        }
    }
}