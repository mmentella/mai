using Xunit;

namespace mai.mnist.Tests
{
    public class DataProviderTests
    {
        [Fact()]
        public void DownloadMNISTTest()
        {
            var mnist = DataProvider.BuildMNIST();
        }
    }
}