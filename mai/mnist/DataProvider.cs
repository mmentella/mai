using mai.blas;

namespace mai.mnist
{
    public static class DataProvider
    {
        public static Uri baseAddress = new("http://yann.lecun.com/exdb/mnist/");

        public static string trainingImagesFilename = "mnist/train-images.idx3-ubyte";
        public static string testImagesFilename = "mnist/t10k-images.idx3-ubyte";
        public static string trainingLabelsFilename = "mnist/train-labels.idx1-ubyte";
        public static string testLabelsFilename = "mnist/t10k-labels.idx1-ubyte";

        public static IEnumerable<(Matrix image, Matrix label)> BuildMNIST()
        {
            using FileStream imageFileStream = new(trainingImagesFilename, FileMode.Open);
            using FileStream labelFileStream = new(trainingLabelsFilename, FileMode.Open);
            using BinaryReader imageReader = new(imageFileStream);
            using BinaryReader labelReader = new(labelFileStream);

            int magicNumber = imageReader.ReadBigInt32();
            int numberOfImages = imageReader.ReadBigInt32();
            int rows = imageReader.ReadBigInt32();
            int columns = imageReader.ReadBigInt32();

            int labelMagic = labelReader.ReadBigInt32();
            int numberOfLabels = labelReader.ReadBigInt32();

            Matrix image;
            Matrix label;
            for (int i = 0; i < numberOfImages; i++)
            {
                image = imageReader.ReadImage(rows, columns);
                label = labelReader.ReadLabel();

                yield return (image, label);
            }
        }

        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static Matrix ReadImage(this BinaryReader br, int rows, int columns)
        {
            var bytes = br.ReadBytes(rows * columns);
            double[] data = bytes.Select(b => (double)b).ToArray();

            return new Matrix(data, 1, rows * columns);
        }

        public static Matrix ReadLabel(this BinaryReader br)
        {
            var label = (int)br.ReadByte();

            double[] data = new double[10];
            data[label] = 1;

            return new(data);
        }
    }
}
