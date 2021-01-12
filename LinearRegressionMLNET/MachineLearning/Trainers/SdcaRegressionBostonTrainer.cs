using LinearRegressionMLNET.MachineLearning.Common;
using LinearRegressionMLNET.MachineLearning.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
namespace LinearRegressionMLNET.MachineLearning.Trainers
{
    public class SdcaRegressionBostonTrainer : TrainerBase
    {
        public SdcaRegressionBostonTrainer() : base()
        {
            Name = "SDCA";
            _model = MlContext.Regression.Trainers
                  .Sdca(labelColumnName: "Label", featureColumnName: "Features");
        }

        protected override EstimatorChain<NormalizingTransformer> BuildDataProcessingPipeline()
        {
            var dataProcessPipeline = MlContext.Transforms
              .CopyColumns("Label", nameof(BostonHousingData.MedianPrice))
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
