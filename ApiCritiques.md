Here are the critiques of the current ML.NET APIs:

# 1. Reading from a text file
```C#
    IDataView trainingData = mlContext.Data.ReadFromTextFile(
        new[]
        {
            new TextLoader.Column("SepalLength", DataKind.R4, 0),
            new TextLoader.Column("SepalWidth", DataKind.R4, 1),
            new TextLoader.Column("PetalLength", DataKind.R4, 2),
            new TextLoader.Column("PetalWidth", DataKind.R4, 3),
            new TextLoader.Column("Label", DataKind.Text, 4),
        },
```

This should just be generated from the `IrisData` class. [Tracking issue](https://github.com/dotnet/machinelearning/issues/561).

# 2. Using Factory Methods over Constructors

https://docs.microsoft.com/en-us/dotnet/standard/design-guidelines/constructor

✓ CONSIDER providing simple, ideally default, constructors.

# 3. MakePredictionFunction requires passing `mlContext`

```C#
    IrisPrediction prediction = model
        .MakePredictionFunction<IrisData, IrisPrediction>(mlContext)
        .Predict(newInput);
```

Potentially we could also have an overload that is just directly "Predict". Then we would have:

```C#
    IrisPrediction prediction = model.Predict<IrisData, IrisPrediction>(newInput);
```

The next step would be just to have a `Train` method on the estimator that directly returned something that can create predictions.

```C#
    var model = estimator.Train<IrisData, IrisPrediction>(trainingData);
    IrisPrediction prediction = model.Predict(newInput);
```

# 4. `MLContext` could be optional

# 5. Setting options should just be properties on the Estimators

Instead of defining a nested "Options" class for every type, we should just have properties on the Estimator classes. This way we don't need so many constructor overloads, and optional parameters. Instead, the constructor only takes the non-optional parameters, and the rest are properties on the type.

_Note: Apache Spark has the same "Estimator/Transformer" concepts, and their Estimators are mutable._
