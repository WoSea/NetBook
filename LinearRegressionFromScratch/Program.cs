/*
 https://rubikscode.net/2021/01/11/machine-learning-with-ml-net-linear-regression/
*/
using System;
using System.Linq;

namespace LinearRegressionFromScratch
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            float[] X = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
            float[] y = { 6, 6, 11, 17, 16, 20, 23, 23, 29, 33, 39 };

            var linearRegressor = new LinearRegressor();
            linearRegressor.Fit(X, y);

            var predictions = linearRegressor.Predict(X);

            Console.WriteLine("Predictions:");
            Console.WriteLine($"{string.Join(", ", predictions.Select(p => p.ToString()))}");

            Console.WriteLine("Actual Value:");
            Console.WriteLine($"{string.Join(", ", y.Select(p => p.ToString()))}");
            Console.ReadLine();
        }
    }
}
