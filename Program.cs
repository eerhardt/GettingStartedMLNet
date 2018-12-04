﻿using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Conversions;
using Microsoft.ML.Trainers;
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
            MLContext mlContext = new MLContext();

            IDataView trainingData = mlContext.Data.ReadFromTextFile(
                typeof(IrisData),
                "iris-data.txt",
                separator: ",");

            var estimator = new ValueToKeyMappingEstimator(mlContext, "Label")
                .Append(new ColumnConcatenatingEstimator(mlContext, "Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(new SdcaMultiClassTrainer(mlContext))
                .Append(new KeyToValueMappingEstimator(mlContext, "PredictedLabel"));

            var model = estimator.Train<IrisData, IrisPrediction>(trainingData);

            IrisData newInput = new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            };
            IrisPrediction prediction = model.Predict(newInput);

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabel}");
        }
    }
}