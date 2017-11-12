using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class BigEndianBinaryReader : BinaryReader
    {     
        private readonly Stream stream;

        public BigEndianBinaryReader(System.IO.Stream stream) : base(stream)
        {
            this.stream = stream;
        }

        public override int ReadInt32()
        {
            var data = base.ReadBytes(4);
            Array.Reverse(data);
            return BitConverter.ToInt32(data, 0);
        }

        public override double ReadDouble()
        {
            var data = base.ReadBytes(8);
            Array.Reverse(data);
            return BitConverter.ToDouble(data, 0);
        }

        public double[][] ReadData(int dataCount)
        {
            var trainingData = new double[dataCount][];

            using (BigEndianBinaryReader reader = new BigEndianBinaryReader(stream))
            {
                for (int i = 0; i < 3; i++)
                {
                    reader.ReadInt32();
                }

                for (int count = 0; count < dataCount; count++)
                {
                    double[] data = new double[Helper.InputDimension];
                    for (int index = 0; index < Helper.InputDimension; index++)
                    {
                        data[index] = reader.ReadDouble();
                    }

                    trainingData[count] = data;
                }
            }

            return trainingData;
        }

        public int[] ReadLabels(int dataCount)
        {
            var labels = new int[dataCount];

            using (BigEndianBinaryReader reader = new BigEndianBinaryReader(stream))
            {
                for (int i = 0; i < 2; i++)
                {
                    reader.ReadInt32();
                }

                for (int count = 0; count < dataCount; count++)
                {
                    labels[count] = reader.ReadByte(); ;
                }
            }

            return labels;
        }
    }
}
