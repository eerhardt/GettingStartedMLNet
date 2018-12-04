using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Conversions;
using Microsoft.ML.Trainers;
using System;

namespace myApp
{
    public class IrisData
    {
        public float SepalLength { get; set; }
        public float SepalWidth { get; set; }
        public float PetalLength { get; set; }
        public float PetalWidth { get; set; }
        public string Label { get; set; }
    }

    public class IrisPrediction
    {
        public string PredictedLabel { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var loader = new TextLoader(typeof(IrisData), separator = ",");

            var trainingData = loader.Read("iris-data.txt");

            var estimator = new ValueToKeyMappingEstimator("Label")
                .Append(new ColumnConcatenatingEstimator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(new SdcaMultiClassTrainer())
                .Append(new KeyToValueMappingEstimator("PredictedLabel"));

            var model = estimator.Train<IrisData, IrisPrediction>(trainingData);

            var newInput = new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            };
            
            var prediction = model.Predict(newInput);

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabel}");
        }
    }
}
