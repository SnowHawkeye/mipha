import _pickle as pickle
import os
import tempfile

import pytest

from src.mipha.framework import MiphaComponent, MiphaPredictor, DataContract, FeatureExtractor


class MockComponent(MiphaComponent):
    def __init__(self, name):
        self.name = name


class MockFeatureExtractor(FeatureExtractor):
    def extract_features(self, x):
        pass

    def __init__(self, name):
        super().__init__(name)
        self.name = name


# noinspection PyTypeChecker
def test_save_and_load():
    # Create mock components
    feature_extractors = [MockFeatureExtractor(name=f"feature_{i}") for i in range(3)]
    aggregator = MockComponent(name="aggregator")
    model = MockComponent(name="model")
    evaluator = MockComponent(name="evaluator")

    mipha_model = MiphaPredictor(feature_extractors, aggregator, model, evaluator)

    # Use a temporary file for the ZIP archive
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = os.path.join(temp_dir, 'model_archive.zip')

        # Test the save function
        mipha_model.save(archive_path)
        assert os.path.exists(archive_path), "Archive file was not created."

        # Test the load function
        loaded_model = MiphaPredictor.load(archive_path)
        assert len(loaded_model.feature_extractors) == len(
            mipha_model.feature_extractors), "Feature extractors were not loaded correctly."
        assert loaded_model.aggregator.name == mipha_model.aggregator.name, "Aggregator was not loaded correctly."
        assert loaded_model.model.name == mipha_model.model.name, "Model was not loaded correctly."
        assert loaded_model.evaluator.name == mipha_model.evaluator.name, "Evaluator was not loaded correctly."


class ExampleFeatureExtractor(FeatureExtractor):
    def extract_features(self, x):
        pass


class AnotherFeatureExtractor(FeatureExtractor):
    def extract_features(self, x):
        pass


@pytest.fixture
def data_contract():
    return DataContract()


def test_add_data_sources_with_valid_mapping(data_contract):
    extractors = [ExampleFeatureExtractor(), AnotherFeatureExtractor()]
    data_contract.add_data_sources({"ExampleDataSourceType": extractors})
    assert data_contract.get_extractors("ExampleDataSourceType") == extractors


def test_add_data_sources_with_single_extractor(data_contract):
    extractor = ExampleFeatureExtractor()
    data_contract.add_data_sources({"ExampleDataSourceType": extractor})
    assert data_contract.get_extractors("ExampleDataSourceType") == [extractor]


def test_add_data_sources_replaces_existing(data_contract, capsys):
    extractors1 = [ExampleFeatureExtractor()]
    extractors2 = [AnotherFeatureExtractor()]
    data_contract.add_data_sources({"ExampleDataSourceType": extractors1})
    data_contract.add_data_sources({"ExampleDataSourceType": extractors2})
    captured = capsys.readouterr()  # captures the print
    assert "Warning" in captured.out
    assert data_contract.get_extractors("ExampleDataSourceType") == extractors2


def test_get_extractors_existing_data_source(data_contract):
    # Adding initial mappings
    extractor1 = ExampleFeatureExtractor()
    extractor2 = AnotherFeatureExtractor()
    data_contract.add_data_sources({
        "source1": [extractor1],
        "source2": [extractor2]
    })

    # Test for existing data source type
    assert data_contract.get_extractors("source1") == [extractor1]
    assert data_contract.get_extractors("source2") == [extractor2]


def test_get_extractors_nonexistent_data_source(data_contract):
    # Adding initial mappings
    extractor1 = ExampleFeatureExtractor()
    data_contract.add_data_sources({
        "source1": [extractor1]
    })

    # Test for nonexistent data source type
    with pytest.raises(KeyError, match="nonexistent_source is not in the contract"):
        data_contract.get_extractors("nonexistent_source")


def test_constructor_with_initial_mappings():
    initial_mappings = {
        "ExampleDataSourceType": [ExampleFeatureExtractor()],
        "AnotherDataSourceType": [AnotherFeatureExtractor()]
    }
    data_contract = DataContract(initial_mappings)
    assert data_contract.get_extractors("ExampleDataSourceType") == initial_mappings["ExampleDataSourceType"]
    assert data_contract.get_extractors("AnotherDataSourceType") == initial_mappings["AnotherDataSourceType"]


def test_from_feature_extractors():
    extractor1 = ExampleFeatureExtractor(managed_data_sources=["source1", "source2"])
    extractor2 = AnotherFeatureExtractor(managed_data_sources=["source2"])

    extractors = [extractor1, extractor2]
    data_contract = DataContract.from_feature_extractors(extractors)

    assert data_contract.get_extractors("source1") == [extractor1]
    assert data_contract.get_extractors("source2") == [extractor1, extractor2]
