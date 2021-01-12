using System.Linq;

namespace LinearRegressionFromScratch
{
    class LinearRegressor
    {
        private float _b0;
        private float _b1;
        public LinearRegressor()
        {
            _b0 = 0;
            _b1 = 0;
        }
        public void Fit(float[] X, float[] y)
        {
            var ssxy = X.Zip(y, (a, b) => a * b).Sum() - X.Length * X.Average() * y.Average();
            var ssxx = X.Zip(X, (a, b) => a * b).Sum() - X.Length * X.Average() * X.Average();

            _b1 = ssxy / ssxx;
            _b0 = y.Average() - _b1 * X.Average();
        }
        public float[] Predict(float[] x)
        {
            return x.Select(i => _b0 + i * _b1).ToArray();
        }

    }
}
