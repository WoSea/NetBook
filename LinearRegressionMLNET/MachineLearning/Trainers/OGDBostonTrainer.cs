using LinearRegressionMLNET.MachineLearning.Common;
using LinearRegressionMLNET.MachineLearning.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
namespace LinearRegressionMLNET.MachineLearning.Trainers
{
    public sealed class OGDBostonTrainer : TrainerBase
    {
        public OGDBostonTrainer() : base()
        {
            Name = "Online Gradient Descent";
            _model = MlContext.Regression.Trainers
          .OnlineGradientDescent(labelColumnName: "Label", featureColumnName: "Features");
        }

        protected override EstimatorChain<NormalizingTransformer> BuildDataProcessingPipeline()
        {
            var dataProcessPipeline = MlContext.Transforms
              .CopyColumns("Label", nameof(BostonHousingData.MedianPrice))
                .Append(MlContext.Transforms.Categorical.OneHotEncoding("RiverCoast"))
                .Append(MlContext.Transforms.Concatenate("Features",
                                                "CrimeRate",
                                                "Zoned",
                                                "Proportion",
                                                "RiverCoast",
                                                "NOConcetration",
                                                "NumOfRoomsPerDwelling",
                                                "Age",
                                                "EmployCenterDistance",
                                                "HighwayAccecabilityRadius",
                                                "TaxRate",
                                                "PTRatio"))
                .Append(MlContext.Transforms.NormalizeLogMeanVariance("Features", "Features"))
                .AppendCacheCheckpoint(MlContext);

            return dataProcessPipeline;
        }
    }
}
