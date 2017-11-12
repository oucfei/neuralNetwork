using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class BackProppagation
    {
        private readonly Random randomDoubleGenerator;

        private int maxNumLoop = 1;

        private double errorTermThreshold = 1e-5;

        public BackProppagation()
        {
            randomDoubleGenerator = new Random();
        }

        public BackPropResult RunBackProp(double[][] trainingData, int[] traingLabels, double learningRate, int hiddenUnitNumber)
        {
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            var firstLayerWeights = new double[Helper.InputDimension][];
            InitWeightMatrix(firstLayerWeights, hiddenUnitNumber, -0.005, 0.005);

            var secondLayerWeights = new double[hiddenUnitNumber][];
            InitWeightMatrix(secondLayerWeights, Helper.OutDimension, -0.005, 0.005); //?

            List<int> loopNumbers = new List<int>();
            for (int index = 0; index < trainingData.Length; index++)
            {
                int loopCounter;
                for (loopCounter = 0; loopCounter < maxNumLoop; loopCounter++)
                {
                    var inputMatrix = ConvertDataToMatrix(trainingData[index]);
                    var hiddenUnitInputs = Helper.MultiplyMatrix(inputMatrix, firstLayerWeights);

                    var hiddenUnitOutputs = GetHiddenUnitOutputs(hiddenUnitInputs);
                    var finalOutputs = Helper.MultiplyMatrix(hiddenUnitOutputs, secondLayerWeights);
                    var outputErrorMatrix = GetErrorTermMatrix(finalOutputs, traingLabels[index]);
                    if (OutputErrorLessThanThreshold(outputErrorMatrix))
                    {                        
                        Console.WriteLine($"All Error less than Threshold. Data index: {index}, Loop counter: {loopCounter}");
                        break;
                    }

                    var hiddenUnitsErrorMatrix =
                        GetHiddenUnitsErrorMatrix(hiddenUnitOutputs, secondLayerWeights, outputErrorMatrix);

                    UpdateWeights(firstLayerWeights, hiddenUnitsErrorMatrix, learningRate, inputMatrix);
                    UpdateWeights(secondLayerWeights, outputErrorMatrix, learningRate, hiddenUnitOutputs);
                }

                loopNumbers.Add(loopCounter);

                if (index >= 10000 && index % 10000 == 0)
                {
                    Console.WriteLine($"Number of training data processed: {index}");
                }
            }

            return new BackPropResult()
            {
                FirstLayerWeights = firstLayerWeights,
                SecondLayerWeights = secondLayerWeights,
                NumberOfLoopsPerData = loopNumbers,
                RunningTimeInSeconds = stopwatch.ElapsedMilliseconds / 1000
            };
        }

        public void ValidateResult(BackPropResult network, double[][] testData, int[] testLaebl)
        {
            int hit = 0;
            for (int index = 0; index < testData.Length; index++)
            {
                var inputMatrix = ConvertDataToMatrix(testData[index]);
                var hiddenUnitInputs = Helper.MultiplyMatrix(inputMatrix, network.FirstLayerWeights);

                var hiddenUnitOutputs = GetHiddenUnitOutputs(hiddenUnitInputs);
                var outputMatrix = Helper.MultiplyMatrix(hiddenUnitOutputs, network.SecondLayerWeights);

                int finalOutput = GetFinalOutputFromMatrix(outputMatrix);
                if (testLaebl[index] == finalOutput)
                {
                    hit++;
                }
            }

            Console.WriteLine($"number of hit: {hit}");
        }

        private int GetFinalOutputFromMatrix(double[][] outputs)
        {
            double minNum = Double.MinValue;
            int index = -1;
            for (int i = 0; i < outputs[0].Length; i++)
            {
                if (outputs[0][i] > minNum)
                {
                    minNum = outputs[0][i];
                    index = i;
                }
            }

            if (index < 0)
            {
                throw new Exception();
            }

            return index;
        }

        private bool OutputErrorLessThanThreshold(double[][] outputErrors)
        {
            bool lessThanThreshold = true;
            for (int i = 0; i < outputErrors[0].Length; i++)
            {
                if (Math.Abs(outputErrors[0][i]) > errorTermThreshold)
                {
                    lessThanThreshold = false;
                }
            }

            return lessThanThreshold;
        }

        private void UpdateWeights(double[][] weightMatrix, double[][] errorMatrix, double learningRate, double[][] inputs)
        {
            for (int i = 0; i < weightMatrix.Length; i++)
            {
                for (int j = 0; j < weightMatrix[0].Length; j++)
                {
                    var weightDiff = GetSafeDoubleValue(learningRate * errorMatrix[0][j] * inputs[0][i]);
                    weightMatrix[i][j] = weightMatrix[i][j] + weightDiff;
                    if (double.IsNaN(weightMatrix[i][j]) || double.IsInfinity(weightMatrix[i][j]))
                    {
                        throw new Exception();
                    }
                }
            }
        }

        private double[][] GetHiddenUnitsErrorMatrix(double[][] hiddenUnitOutputs, double[][] secondLayerWeights, double[][] outputErrorMatrix)
        {
            var errorMatrix = new double[1][];
            errorMatrix[0] = new double[hiddenUnitOutputs[0].Length];

            for (int i = 0; i < errorMatrix[0].Length; i++)
            {
                var outputByWeight = 0.0;
                for(int k = 0; k < outputErrorMatrix[0].Length; k++)
                {
                    outputByWeight += GetSafeDoubleValue(outputErrorMatrix[0][k] * secondLayerWeights[i][k]);
                }

                errorMatrix[0][i] = GetSafeDoubleValue(hiddenUnitOutputs[0][i] * (1 - hiddenUnitOutputs[0][i]) * outputByWeight);

                if (double.IsNaN(errorMatrix[0][i]) || double.IsInfinity(errorMatrix[0][i]))
                {
                    throw new Exception();
                }
            }

            return errorMatrix;
        }

        private double[][] GetErrorTermMatrix(double[][] networkOutputs, int expectedValue)
        {
            var errorMatrix = new double[1][];
            errorMatrix[0] = new double[networkOutputs[0].Length];
            for (int i = 0; i < networkOutputs[0].Length; i++)
            {
                double output = networkOutputs[0][i];
                if (expectedValue == i)
                {
                    errorMatrix[0][i] = GetSafeDoubleValue(output * (1 - output) * (1 - output));
                }
                else
                {
                    errorMatrix[0][i] = GetSafeDoubleValue(output * (1 - output));
                }

                if (double.IsNaN(errorMatrix[0][i]) || double.IsInfinity(errorMatrix[0][i]))
                {
                    throw new Exception();
                }
            }

            return errorMatrix;
        }

        private double[][] GetHiddenUnitOutputs(double[][] inputs)
        {
            var hiddenUnitOutputs = new double[1][];
            hiddenUnitOutputs[0] = new double[inputs[0].Length];

            for (int i = 0; i < inputs[0].Length; i++)
            {
                hiddenUnitOutputs[0][i] = Sigmoid(inputs[0][i]);
            }

            return hiddenUnitOutputs;
        }

        private double Sigmoid(double x)
        {
            return GetSafeDoubleValue(1.0 / (1 + Math.Exp(-x)));
        }

        private double[][] ConvertDataToMatrix(double[] data)
        {
            var matrix = new double[1][];
            matrix[0] = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                matrix[0][i] = data[i];
            }

            return matrix;
        }

        private void InitWeightMatrix(double[][] weightMatrix, int colSize, double minimum, double maximum)
        {
            for (int row = 0; row < weightMatrix.Length; row++)
            {
                weightMatrix[row] = new double[colSize];
            }

            for (int i = 0; i < weightMatrix.Length; i++)
            {
                for (int j = 0; j < weightMatrix[0].Length; j++)
                {
                    weightMatrix[i][j] = randomDoubleGenerator.NextDouble() * (maximum - minimum) + minimum;
                }
            }
        }

        private double GetSafeDoubleValue(double value)
        {
            if (double.IsNegativeInfinity(value))
            {
                return double.MinValue;
            }

            if (double.IsPositiveInfinity(value))
            {
                return double.MaxValue;
            }
            
            return value;
        }
    }
}
