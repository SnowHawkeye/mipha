from typing import Callable, Type

from mipha.framework import FeatureExtractor, Aggregator


def make_feature_extractor(extract_features: Callable) -> Type[FeatureExtractor]:
    """
    :param extract_features: The feature extraction method used by the FeatureExtractor implementation.
    :return: A simple implementation of FeatureExtractor.
    """

    class FeatureExtractorImplementation(FeatureExtractor):
        def extract_features(self, x):
            return extract_features(x)

    return FeatureExtractorImplementation


def make_feature_aggregator(aggregate_features: Callable) -> Type[Aggregator]:
    """
    :param aggregate_features: The feature aggregation method used by the Aggregator implementation.
    :return: A simple implementation of Aggregator.
    """

    class AggregatorImplementation(Aggregator):
        def aggregate_features(self, data):
            return aggregate_features(data)

    return AggregatorImplementation
