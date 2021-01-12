using System;

namespace MultipleLinearRegressionFromScratch
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            double[,] X = { { 1, 2, 3},
                            { 2, 9, 11},
                            { 56, 111, 66}};

            double[,] y = { { 6 }, { 6 }, { 11 } };

            var linearRegressor = new MultipleLinearRegressor();
            linearRegressor.Fit(X, y);

            var prediction = linearRegressor.Predict(new double[,] { { 3 }, { 5 }, { 7 } });

            Console.WriteLine($"Prediction: {prediction}");
        }
    }
}
