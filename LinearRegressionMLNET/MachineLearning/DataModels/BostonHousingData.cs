using Microsoft.ML.Data;

namespace LinearRegressionMLNET.MachineLearning.DataModels
{
    class BostonHousingData
    {
        [LoadColumn(0)]
        public float CrimeRate { get; set; }

        [LoadColumn(1)]
        public float Zoned { get; set; }

        [LoadColumn(2)]
        public float Proportion { get; set; }

        [LoadColumn(3)]
        public float RiverCoast { get; set; }

        [LoadColumn(4)]
        public float NOConcetration { get; set; }

        [LoadColumn(5)]
        public float NumOfRoomsPerDwelling { get; set; }

        [LoadColumn(6)]
        public float Age { get; set; }

        [LoadColumn(7)]
        public float EmployCenterDistance { get; set; }

        [LoadColumn(8)]
        public float HighwayAccecabilityRadius { get; set; }

        [LoadColumn(9)]
        public float TaxRate { get; set; }

        [LoadColumn(10)]
        public float PTRatio { get; set; }

        [LoadColumn(11)]
        public float MedianPrice { get; set; }
    }
}
