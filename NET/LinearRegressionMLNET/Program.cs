using LinearRegressionMLNET.MachineLearning.Common;
using LinearRegressionMLNET.MachineLearning.DataModels;
using LinearRegressionMLNET.MachineLearning.Predictors;
using LinearRegressionMLNET.MachineLearning.Trainers;
using System;

namespace LinearRegressionMLNET
{
    class Program
    {
        static void Main(string[] args)
        {
            var newSample = new BostonHousingData
            {
                Age = 65.2f,
                CrimeRate = 0.00632f,
                EmployCenterDistance = 4.0900f,
                HighwayAccecabilityRadius = 15.3f,
                NOConcetration = 0.538f,
                NumOfRoomsPerDwelling = 6.575f,
                Proportion = 1f,
                PTRatio = 15.3f,
                RiverCoast = 0f,
                TaxRate = 296f,
                Zoned = 23f
            };

            var ogd = new OGDBostonTrainer();
            TrainEvaluatePredict(ogd, newSample);

            var sdca = new SdcaRegressionBostonTrainer();
            TrainEvaluatePredict(sdca, newSample);
        }

        static void TrainEvaluatePredict(TrainerBase trainer, BostonHousingData newSample)
        {
            Console.WriteLine("*******************************");
            Console.WriteLine($"{ trainer.Name }");
            Console.WriteLine("*******************************");

            trainer.Fit("..\\Data\\boston_housing.csv");

            var modelMetrics = trainer.Evaluate();

            Console.WriteLine($"Loss Function: {modelMetrics.LossFunction:0.##}{Environment.NewLine}" +
                              $"Mean Absolute Error: {modelMetrics.MeanAbsoluteError:#.##}{Environment.NewLine}" +
                              $"Mean Squared Error: {modelMetrics.MeanSquaredError:#.##}{Environment.NewLine}" +
                              $"RSquared: {modelMetrics.RSquared:0.##}{Environment.NewLine}" +
                              $"Root Mean Squared Error: {modelMetrics.RootMeanSquaredError:#.##}");

            trainer.Save();

            var predictor = new Predictor();
            var prediction = predictor.Predict(newSample);
            Console.WriteLine("------------------------------");
            Console.WriteLine($"Prediction: {prediction.MedianPrice:#.##}");
            Console.WriteLine("------------------------------");
        }
    }
}
