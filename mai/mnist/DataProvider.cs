using mai.blas;
using System.Diagnostics;

namespace mai.mnist
{
    public static class DataProvider
    {
        public static Uri baseAddress = new("http://yann.lecun.com/exdb/mnist/");

        public static string trainingImagesFilename = "mnist/train-images.idx3-ubyte";
        public static string testImagesFilename = "mnist/t10k-images.idx3-ubyte";
        public static string trainingLabelsFilename = "mnist/train-labels.idx1-ubyte";
        public static string testLabelsFilename = "mnist/t10k-labels.idx1-ubyte";

        public static Matrix ReadSamples(string filename, int maxSamples)
        {
            using FileStream sampleFileStream = new(filename, FileMode.Open);
            using BinaryReader reader = new(sampleFileStream);

            int magicNumber = reader.ReadBigInt32();
            int numberOfImages = reader.ReadBigInt32();
            numberOfImages= Math.Min(numberOfImages, maxSamples);

            int rows = reader.ReadBigInt32();
            int columns = reader.ReadBigInt32();

            float[] image;
            Matrix data = new(numberOfImages, rows * columns);
            for (int i = 0; i < numberOfImages; i++)
            {
                image = reader.ReadImage(rows, columns);
                for (int l = 0; l < rows * columns; l++)
                {
                    data[i, l] = image[l];
                }
                //Debug.WriteLine($"Image loaded {i + 1}");
            }

            return data;
        }

        public static Matrix ReadLabels(string filename, int maxSamples)
        {
            using FileStream sampleFileStream = new(filename, FileMode.Open);
            using BinaryReader reader = new(sampleFileStream);

            int labelMagic = reader.ReadBigInt32();
            int numberOfLabels = reader.ReadBigInt32();
            numberOfLabels=Math.Min(numberOfLabels, maxSamples);

            float[] label;
            Matrix data = new(numberOfLabels, 10);
            for (int i = 0; i < numberOfLabels; i++)
            {
                label = reader.ReadLabel();
                for (int l = 0; l < 10; l++)
                {
                    data[i, l] = label[l];
                }
            }

            return data;
        }

        public static (Matrix samples, Matrix labels, Matrix testSamples, Matrix testLabels) BuildMNIST(int maxSamples)
        {
            Matrix samples = ReadSamples(trainingImagesFilename, maxSamples);
            Matrix labels = ReadLabels(trainingLabelsFilename, maxSamples);
            Matrix testSamples = ReadSamples(testImagesFilename, maxSamples);
            Matrix testLabels = ReadLabels(testLabelsFilename, maxSamples);

            return (samples, labels, testSamples, testLabels);
        }

        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static float[] ReadImage(this BinaryReader br, int rows, int columns)
        {
            var bytes = br.ReadBytes(rows * columns);
            float[] data = bytes.Select(b => (float)b).ToArray();

            return data;
        }

        public static float[] ReadLabel(this BinaryReader br)
        {
            var label = (int)br.ReadByte();

            float[] data = new float[10];
            data[label] = 1;

            return data;
        }
    }
}
