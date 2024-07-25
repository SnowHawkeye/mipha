import numpy as np
import pytest

from tests.test_dummy_data.dummy_implementation import DummyFeatureExtractor, DummyAggregator, DummyMachineLearningModel


@pytest.fixture
def dummy_feature_extractor():
    return DummyFeatureExtractor(component_name="TestFeatureExtractor")


# Feature extractor
def test_extract_features_initial_call(dummy_feature_extractor):
    result = dummy_feature_extractor.extract_features(10)

    assert dummy_feature_extractor.called == 1
    assert result == 10


def test_extract_features_multiple_calls(dummy_feature_extractor):
    result1 = dummy_feature_extractor.extract_features(20)
    assert dummy_feature_extractor.called == 1
    assert result1 == 20

    result2 = dummy_feature_extractor.extract_features(30)
    assert dummy_feature_extractor.called == 2
    assert result2 == 30

    result3 = dummy_feature_extractor.extract_features(40)
    assert dummy_feature_extractor.called == 3
    assert result3 == 40


# Aggregator
def test_aggregate_features_single_array():
    aggregator = DummyAggregator()
    features = [np.array([1, 2, 3])]
    result = aggregator.aggregate_features(features)

    assert aggregator.called == 1
    assert np.array_equal(result, np.array([1, 2, 3]))


def test_aggregate_features_multiple_arrays():
    aggregator = DummyAggregator()
    features = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    result = aggregator.aggregate_features(features)

    assert aggregator.called == 1
    assert np.array_equal(result, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))


def test_aggregate_features_multiple_calls():
    aggregator = DummyAggregator()
    features1 = [np.array([1, 2]), np.array([3, 4])]
    features2 = [np.array([5, 6]), np.array([7, 8])]

    result1 = aggregator.aggregate_features(features1)
    assert aggregator.called == 1
    assert np.array_equal(result1, np.array([1, 2, 3, 4]))

    result2 = aggregator.aggregate_features(features2)
    assert aggregator.called == 2
    assert np.array_equal(result2, np.array([5, 6, 7, 8]))


# Machine learning model
def test_fit_initial_call():
    model = DummyMachineLearningModel()
    x_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    model.fit(x_train, y_train)

    assert model.called_fit == 1


def test_fit_multiple_calls():
    model = DummyMachineLearningModel()
    x_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])

    model.fit(x_train, y_train)
    assert model.called_fit == 1

    model.fit(x_train, y_train)
    assert model.called_fit == 2

    model.fit(x_train, y_train)
    assert model.called_fit == 3


def test_predict():
    model = DummyMachineLearningModel()
    x_test = np.array([[5, 6], [7, 8]])
    result = model.predict(x_test)

    assert result == 1
