using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    class Program
    {
        private const double LearningRate = 0.1;

        private const int HiddenUnitNumber = 10;

        static void Main(string[] args)
        {
            double[][] trainingData;
            int[] trainingLabel;
            double[][] testData;
            int[] testLabel;

            using (var fs = File.OpenRead("E:\\MachineLearning\\HW3\\MNIST_PCA\\train-images-pca.idx2-double"))
            {
                var dataReader = new BigEndianBinaryReader(fs);
                trainingData = dataReader.ReadData(Helper.TrainingDataCount);
            }

            using (var fs = File.OpenRead("E:\\MachineLearning\\HW3\\MNIST_PCA\\train-labels.idx1-ubyte"))
            {
                var dataReader = new BigEndianBinaryReader(fs);
                trainingLabel = dataReader.ReadLabels(Helper.TrainingDataCount);
            }

            using (var fs = File.OpenRead("E:\\MachineLearning\\HW3\\MNIST_PCA\\train-images-pca.idx2-double"))
            {
                var dataReader = new BigEndianBinaryReader(fs);
                testData = dataReader.ReadData(Helper.TestDataCount);
            }

            using (var fs = File.OpenRead("E:\\MachineLearning\\HW3\\MNIST_PCA\\train-labels.idx1-ubyte"))
            {
                var dataReader = new BigEndianBinaryReader(fs);
                testLabel = dataReader.ReadLabels(Helper.TestDataCount);
            }

            var backProp = new BackProppagation();
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            var result = backProp.RunBackProp(trainingData, trainingLabel, LearningRate, HiddenUnitNumber);

            backProp.ValidateResult(result, testData, testLabel);
            Console.WriteLine(stopwatch.ElapsedMilliseconds);
        }
    }
}
