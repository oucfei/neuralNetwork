using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public static class Helper
    {
        public static int TrainingDataCount = 60000;

        public static int TestDataCount = 10000;

        public static int InputDimension = 50;

        public static int OutDimension = 10;

        public static double[][] MultiplyMatrix(double[][] A, double[][] B)
        {
            int rA = A.Length;
            int cA = A[0].Length;
            int rB = B.Length;
            int cB = B[0].Length;
            double[][] kHasil = new double[rA][];
            if (cA != rB)
            {
                throw new Exception("matrix can't be multiplied !!");
            }

            for (int i = 0; i < rA; i++)
            {
                kHasil[i] = new double[cB];
                for (int j = 0; j < cB; j++)
                {
                    double temp = 0;
                    for (int k = 0; k < cA; k++)
                    {
                        temp += A[i][k] * B[k][j];
                    }
                    kHasil[i][j] = temp;
                }
            }
            return kHasil;          
        }
    }
}
