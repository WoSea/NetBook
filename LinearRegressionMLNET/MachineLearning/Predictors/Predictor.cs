using LinearRegressionMLNET.MachineLearning.DataModels;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
namespace LinearRegressionMLNET.MachineLearning.Predictors
{
    public class Predictor
    {
        protected static string ModelPath =>
                Path.Combine(AppContext.BaseDirectory, "regression.mdl");
        private readonly MLContext _mlContext;

        private ITransformer _model;

        public Predictor()
        {
            _mlContext = new MLContext(111);
        }

        /// <summary>
        /// Runs prediction on new data.
        /// </summary>
        /// <param name="newSample">New data sample.</param>
        /// <returns>Predictions made by model.</returns>
        public BostonHousingPricePredictions Predict(BostonHousingData newSample)
        {
            LoadModel();

            var predictionEngine = _mlContext.Model
          .CreatePredictionEngine<BostonHousingData, BostonHousingPricePredictions>(_model);

            return predictionEngine.Predict(newSample);
        }

        private void LoadModel()
        {
            if (!File.Exists(ModelPath))
            {
                throw new FileNotFoundException($"File {ModelPath} doesn't exist.");
            }

            using (var stream = new FileStream(
                        ModelPath,
                            FileMode.Open,
                        FileAccess.Read,
                        FileShare.Read))
            {
                _model = _mlContext.Model.Load(stream, out _);
            }

            if (_model == null)
            {
                throw new Exception($"Failed to load Model");
            }
        }
    }
}
