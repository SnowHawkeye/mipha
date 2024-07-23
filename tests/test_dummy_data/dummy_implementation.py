from src.mipha.framework import *
import numpy as np


class DummyFeatureExtractor(FeatureExtractor):
    def __init__(self, component_name, managed_data_types=None):
        super().__init__(component_name=component_name, managed_data_types=managed_data_types)
        self.called = 0

    def extract_features(self, x):
        self.called += 1
        print(f"extract_features called in {self.component_name}")
        print(f"{self.component_name} was called {self.called} times\n")
        return x


class DummyAggregator(Aggregator):
    def __init__(self):
        super().__init__()
        self.called = 0

    def aggregate_features(self, features):
        self.called += 1
        print(f"Features aggregated from {len(features)} sources")
        print(f"aggregator was called {self.called} times\n")
        return np.hstack(features)


class DummyMachineLearningModel(MachineLearningModel):
    def __init__(self):
        super().__init__()
        self.called_fit = 0

    def fit(self, x_train, y_train, *args, **kwargs):
        self.called_fit += 1
        print("machine learning model fit!")
        print(f"this model was fit {self.called_fit} times")

    def predict(self, x_test, *args, **kwargs):
        print("predict called in Machine Learning Model")
        return 1


class DummyEvaluator(Evaluator):
    def evaluate_model(self, model: DummyMachineLearningModel, x_test, y_test, *args, **kwargs):
        print(f"evaluate_model called for x_test of length {len(x_test)} and y_test of length {len(y_test)}")
