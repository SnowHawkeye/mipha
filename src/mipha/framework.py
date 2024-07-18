from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    """
    Feature Extractor base class. The purpose of a Feature Extractor is to output features from a data source.
    """

    @abstractmethod
    def extract_features(self, x):
        """
        :param x: Input data whose features should be extracted.
        :return: Extracted features. Must be able to cast to a numpy array.
        """
        pass


class Aggregator(ABC):
    """
    Aggregator base class. The purpose of an Aggregator is to combine features from different Feature Extractors
    in order to create matrices usable by a machine learning model.
    """

    @abstractmethod
    def aggregate_features(self, features):
        """
        :param features: List of features to be combined.
        :return: Aggregated features. Must be able to cast to a numpy array.
        """
        pass


class MachineLearningModel(ABC):
    """
    Wrapper for machine learning models used within the MIPHA framework.
    """

    def __init__(self):
        self.model = None  # guarantees the existence of the attribute

    @abstractmethod
    def fit(self, x_train, y_train, *args, **kwargs):
        return self.model.fit(x_train, y_train, *args, **kwargs)

    @abstractmethod
    def predict(self, x_test, *args, **kwargs):
        return self.model.predict(x_test, *args, **kwargs)


class Evaluator(ABC):
    """
    Evaluator base class. Its implementations can customize how the model's performance is evaluated.
    """

    @abstractmethod
    def evaluate_model(self, model: MachineLearningModel, x_test, y_test, *args, **kwargs):
        pass


class MiphaPredictor(ABC):
    """
    Main class of the Modular data Integration for Predictive Healthcare Analysis (MIPHA) model.

    Its components are:
     - Feature Extractors: Each Feature Extractor will be used to extract features from a data source. They must be provided in the same order as their corresponding data sources.
     - Aggregator: The Aggregator will be used to combine features from output by the Feature Extractors.
     - Machine Learning Model: The prediction model itself. The object is a wrapper allowing for the customization of the data processing (e.g. imputation, scaling, etc.).
     - Evaluator: The Evaluator will be used to evaluate the prediction model's performance.

    By analogy with the scikit-learn API, a MIPHA model exposes a `fit` and `predict` methods.
    """

    def __init__(self,
                 feature_extractors: list[FeatureExtractor],
                 aggregator: Aggregator,
                 model: MachineLearningModel,
                 evaluator: Evaluator,
                 ):
        self.feature_extractors = feature_extractors
        self.aggregator = aggregator
        self.model = model
        self.evaluator = evaluator

    def process_data(self, data_sources: list):
        """Utility function to process the data sources by extracting and aggregating features."""
        print("Extracting features from data sources...")
        features = [extractor.extract_features(data_source)
                    for data_source, extractor in zip(data_sources, self.feature_extractors)]
        print("Feature extraction complete!\n")

        print("Aggregating features from data sources...")
        aggregation = self.aggregator.aggregate_features(features)
        print("Aggregation complete!\n")

        return aggregation

    def fit(self, data_sources: list, train_labels, *args, **kwargs):
        """Fit the MIPHA model to the given data sources ("x_train") using the provided training labels ("y_train")."""
        print("Fitting the model...")
        x_train = self.process_data(data_sources)

        output = self.model.fit(x_train, train_labels, *args, **kwargs)
        print("Model fit successfully!\n")

        return output

    def predict(self, data_sources: list, *args, **kwargs):
        """Use the MIPHA model to predict a label for the given data sources ("x_test")."""
        x_test = self.process_data(data_sources)
        return self.model.predict(x_test, *args, **kwargs)

    def evaluate(self, data_sources, test_labels, *args, **kwargs):
        """Evaluate the MIPHA model on the given data sources ("x_test"),
        using the provided test labels ("y_test") as reference."""
        x_test = self.process_data(data_sources)
        return self.evaluator.evaluate_model(self.model, x_test, test_labels, *args, **kwargs)
