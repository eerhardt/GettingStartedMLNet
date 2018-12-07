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

# 1.a Inferring column numbers
Don't need to explicitly state the ordinal

# 2. Building EstimatorChains/Pipelines

There are a few issues with the following code:

```C#
    var estimator = mlContext.Transforms.Conversion.MapValueToKey("Label")
        .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
        .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent())
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
```

1. What exactly is `MLContext`, and why does every Estimator need one at construction time?
    1. What happens if I had 2 `MLContext` objects, and constructed an EstimatorChain using Estimators from both contexts? Is that an error case?
    2. `MLContext` seems more to be about execution, not necessarily tied to a full life-time of an object.
    3. Basically to do anything in the API requires an `MLContext`, which isn't a pattern in most low-level .NET libraries. For example, if I want to extract a method that creates an EstimatorChain, now I need an `MLContext` to be passed to this method.
2. Creating estimator objects this way (factory methods off of `MLContext` instance) doesn't conform to "standard .NET" APIs for creating objects. Instead, we typically use constructors, or static factory methods.
3. Since Estimators require a considerable number of parameters, we invented a whole new mutable object (Arguments) to hold all the parameters. Instead, a more "standard .NET" approach would be to create the Estimator and set properties on it.

Following these issues, we propose changing Estimators to be more typical .NET objects, that can be constructed, have properties set on them as usual. They don't require an `MLContext` in order to be constructed. Instead, they can be created like "typical" .NET objects.

An `MLContext` should only be necessary when you call `Fit()`, and even then we can make it optional. If a user doesn't require any specific context settings, they don't need to provide one.

## Proposed API

### Proposal #1

Using simple constructors. This has the advantage of being more natural, and allows for Object Initialization patterns:

```C#
    var estimator = new ValueToKeyMappingEstimator("Label")
        .Append(new ColumnConcatenatingEstimator("Features", new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" }))
        .Append(new SdcaMultiClassTrainer()
        {
            BiasLearningRate = 0.5f
        })
        .Append(new KeyToValueMappingEstimator("PredictedLabel"));
```

### Proposal #2

Using static factory methods. This has the advantage of being able to catalog which kinds of estimators are available - instead of being able to `new` up any class in scope. The users only see the transforms that can be created.

This approach has the disadvantage to the current approach that you cannot have extension static methods. The current approach allows for the static methods to be extended when you add a new NuGet package.

```C#
    var estimator = Transforms.MapValueToKey("Label")
        .Append(Transforms.Concatenate("Features", new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" }))
        .Append(MulticlassClassificationTrainers.StochasticDualCoordinateAscent())
        .Append(Transforms.MapKeyToValue("PredictedLabel"));
```

# Appendix
https://docs.microsoft.com/en-us/dotnet/standard/design-guidelines/constructor

✓ CONSIDER providing simple, ideally default, constructors.

✓ CONSIDER using a static factory method instead of a constructor if the semantics of the desired operation do not map directly to the construction of a new instance, or if following the constructor design guidelines feels unnatural.

## Mutable Estimators

Instead of defining a nested "Options" class for every type, we should just have properties on the Estimator classes. This way we don't need so many constructor overloads, and optional parameters. Instead, the constructor only takes the non-optional parameters, and the rest are properties on the type.

_Note: Apache Spark has the same "Estimator/Transformer" concepts, and their Estimators are mutable._
