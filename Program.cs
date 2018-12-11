using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Conversions;
using System;

namespace myApp
{
    class Program
    {
        public class IrisData
        {
            [Column(0)]
            public float SepalLength { get; set; }
            [Column(1)]
            public float SepalWidth { get; set; }
            [Column(2)]
            public float PetalLength { get; set; }
            [Column(3)]
            public float PetalWidth { get; set; }
            [Column(4)]
            public string Label { get; set; }
        }

        public class IrisPrediction
        {
            public string PredictedLabel { get; set; }
        }

        static void Main(string[] args)
        {
            TextLoader loader = TextLoader.Create(typeof(IrisData), separator: ",");
            IDataView trainingData = loader.Read("iris-data.txt");

            var pipeline = new EstimatorChain();
            pipeline.Add(new ValueToKeyMappingEstimator(nameof(IrisData.Label)));
            pipeline.Add(new ColumnConcatenatingEstimator(
                inputColumns: new[] 
                {
                    nameof(IrisData.SepalLength),
                    nameof(IrisData.SepalWidth),
                    nameof(IrisData.PetalLength),
                    nameof(IrisData.PetalWidth)
                },
                outputColumn: DefaultColumnNames.Features));
            pipeline.Add(new SdcaMultiClassTrainer(
                featureColumn: DefaultColumnNames.Features,
                labelColumn: nameof(IrisData.Label),
                predictedLabelColumn: nameof(IrisPrediction.PredictedLabel)));
            pipeline.Add(new KeyToValueMappingEstimator(nameof(IrisPrediction.PredictedLabel)));

            var model = pipeline.Fit(trainingData);

            IrisData newInput = new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            };
            IrisPrediction prediction = model
                .MakePredictionFunction<IrisData, IrisPrediction>()
                .Predict(newInput);

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabel}");
        }
    }
}