using MathNet.Numerics.LinearAlgebra;
using System;
using System.Linq;

namespace MultipleLinearRegressionFromScratch
{
    class MultipleLinearRegressor
    {
        private double _b;
        private double[] _w;

        public MultipleLinearRegressor()
        {
            _b = 0;
        }

        public void Fit(double[,] X, double[,] y)
        {
            var input = ExtendInputWithOnes(X);
            var output = Matrix<double>.Build.DenseOfArray(y);

            var coeficients = ((input.Transpose() * input).Inverse() * input.Transpose() * output)
                           .Transpose().Row(0);
            _b = coeficients.ElementAt(0);
            _w = SubArray(coeficients.ToArray(), 1, X.GetLength(1));
        }

        public double Predict(double[,] x)
        {
            var input = Matrix<double>.Build.DenseOfArray(x).Transpose();
            var w = Vector<double>.Build.DenseOfArray(_w);
            return input.Multiply(w).ToArray().Sum() + _b;
        }

        private Matrix<double> ExtendInputWithOnes(double[,] X)
        {
            // Add 'ones' to the input array to model coefficient b in data.
            var ones = Matrix<double>.Build.Dense(X.GetLength(0), 1, 1d);
            var extendedX = ones.Append(Matrix<double>.Build.DenseOfArray(X));

            return extendedX;
        }

        private double[] SubArray(double[] data, int index, int length)
        {
            double[] result = new double[length];
            Array.Copy(data, index, result, 0, length);
            return result;
        }
    }
}
