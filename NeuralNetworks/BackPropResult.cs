using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class BackPropResult
    {
        public double[][] FirstLayerWeights { get; set; }

        public double[][] SecondLayerWeights { get; set; }

        public long RunningTimeInSeconds { get; set; }

        public List<int> NumberOfLoopsPerData { get; set; }

    }
}
