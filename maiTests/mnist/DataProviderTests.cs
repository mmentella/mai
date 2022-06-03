using Xunit;
using mai.mnist;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using mai.blas;

namespace mai.mnist.Tests
{
    public class DataProviderTests
    {
        [Fact()]
        public void DownloadMNISTTest()
        {
            IEnumerable<(Matrix image, Matrix label)> trainingSet = DataProvider.BuildMNIST();
            var array = trainingSet.ToArray();
        }
    }
}