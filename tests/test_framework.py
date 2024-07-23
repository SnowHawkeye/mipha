import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.mipha.framework import MiphaComponent, MiphaPredictor, DataContract, FeatureExtractor


# Save and load tests #############################################

class MockComponent(MiphaComponent):
    def __init__(self, component_name):
        super().__init__(component_name)


class MockFeatureExtractor(FeatureExtractor):
    def extract_features(self, x):
        pass

    def __init__(self, component_name):
        super().__init__(component_name)


# noinspection PyTypeChecker,PyUnresolvedReferences
def test_save_and_load():
    # Create mock components
    feature_extractors = [MockFeatureExtractor(component_name=f"feature_{i}") for i in range(3)]
    aggregator = MockComponent(component_name="aggregator")
    model = MockComponent(component_name="model")
    evaluator = MockComponent(component_name="evaluator")

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
        assert loaded_model.aggregator.component_name == mipha_model.aggregator.component_name, "Aggregator was not loaded correctly."
        assert loaded_model.model.component_name == mipha_model.model.component_name, "Model was not loaded correctly."
        assert loaded_model.evaluator.component_name == mipha_model.evaluator.component_name, "Evaluator was not loaded correctly."


# DataContract tests #############################################

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
    with pytest.raises(KeyError, match="nonexistent_source data source type is not in the contract"):
        data_contract.get_extractors("nonexistent_source")


def test_data_contract_constructor_with_initial_mappings():
    initial_mappings = {
        "ExampleDataSourceType": [ExampleFeatureExtractor()],
        "AnotherDataSourceType": [AnotherFeatureExtractor()]
    }
    data_contract = DataContract(initial_mappings)
    assert data_contract.get_extractors("ExampleDataSourceType") == initial_mappings["ExampleDataSourceType"]
    assert data_contract.get_extractors("AnotherDataSourceType") == initial_mappings["AnotherDataSourceType"]


def test_from_feature_extractors():
    extractor1 = ExampleFeatureExtractor(managed_data_types=["source1", "source2"])
    extractor2 = AnotherFeatureExtractor(managed_data_types=["source2"])

    extractors = [extractor1, extractor2]
    data_contract = DataContract.from_feature_extractors(extractors)

    assert data_contract.get_extractors("source1") == [extractor1]
    assert data_contract.get_extractors("source2") == [extractor1, extractor2]


# Data processing tests #############################################

@pytest.fixture
def feature_extractor_mock():
    extractor = MagicMock()
    extractor.component_name = 'extractor_1'
    extractor.extract_features.return_value = 'features_1'
    return extractor


@pytest.fixture
def aggregator_mock():
    aggregator = MagicMock()
    aggregator.aggregate_features.return_value = 'aggregated_features'
    return aggregator


@pytest.fixture
def data_source_mock():
    data_source = MagicMock()
    data_source.data_type = 'type_1'
    data_source.name = 'data_source_1'
    data_source.data = 'data'
    return data_source


@pytest.fixture
def data_contract_mock(feature_extractor_mock):
    data_contract = MagicMock()
    data_contract.get_extractors.return_value = [feature_extractor_mock]
    return data_contract


@pytest.fixture
def mipha_predictor(feature_extractor_mock, aggregator_mock, data_contract_mock):
    with patch('src.mipha.framework.DataContract.from_feature_extractors', return_value=data_contract_mock):
        return MiphaPredictor([feature_extractor_mock, feature_extractor_mock], aggregator_mock, MagicMock(),
                              MagicMock())


def test_process_data_no_precomputed(mipha_predictor, data_source_mock, feature_extractor_mock):
    data_sources = [data_source_mock]
    expected_features = 'features_1'
    expected_last_computed_features = {('data_source_1', 'extractor_1'): expected_features}

    result = mipha_predictor.process_data(data_sources)

    feature_extractor_mock.extract_features.assert_called_once_with(data_source_mock.data)
    mipha_predictor.aggregator.aggregate_features.assert_called_once_with([expected_features])

    assert mipha_predictor.last_computed_features == expected_last_computed_features
    assert result == 'aggregated_features'


def test_process_data_with_precomputed(mipha_predictor, data_source_mock, feature_extractor_mock):
    data_sources = [data_source_mock]
    precomputed_features = {('data_source_1', 'extractor_1'): 'precomputed_features'}
    expected_last_computed_features = {('data_source_1', 'extractor_1'): 'precomputed_features'}

    result = mipha_predictor.process_data(data_sources, precomputed_features)

    feature_extractor_mock.extract_features.assert_not_called()
    mipha_predictor.aggregator.aggregate_features.assert_called_once_with(['precomputed_features'])

    assert mipha_predictor.last_computed_features == expected_last_computed_features
    assert result == 'aggregated_features'


def test_process_data_with_partial_precomputed(mipha_predictor, data_source_mock, feature_extractor_mock):
    data_sources = [data_source_mock]
    precomputed_features = {('data_source_1', 'extractor_1'): 'precomputed_features'}

    # Mock another extractor
    feature_extractor_mock_2 = MagicMock()
    feature_extractor_mock_2.component_name = 'extractor_2'
    feature_extractor_mock_2.extract_features.return_value = 'features_2'

    mipha_predictor.data_contract.get_extractors.return_value = [feature_extractor_mock, feature_extractor_mock_2]

    result = mipha_predictor.process_data(data_sources, precomputed_features)

    expected_last_computed_features = {
        ('data_source_1', 'extractor_1'): 'precomputed_features',
        ('data_source_1', 'extractor_2'): 'features_2'
    }

    feature_extractor_mock.extract_features.assert_not_called()
    feature_extractor_mock_2.extract_features.assert_called_once_with(data_source_mock.data)
    mipha_predictor.aggregator.aggregate_features.assert_called_once_with(['precomputed_features', 'features_2'])

    assert mipha_predictor.last_computed_features == expected_last_computed_features
    assert result == 'aggregated_features'


# Fit and predict tests #############################################

def test_fit_no_precomputed(mipha_predictor, data_source_mock):
    data_sources = [data_source_mock]
    train_labels = [1, 0, 1]

    with patch.object(mipha_predictor, 'process_data', return_value='processed_data') as process_data_mock:
        with patch.object(mipha_predictor.model, 'fit', return_value='model_output') as model_fit_mock:
            result = mipha_predictor.fit(data_sources, train_labels)

            process_data_mock.assert_called_once_with(data_sources, None)
            model_fit_mock.assert_called_once_with('processed_data', train_labels)

            assert result == 'model_output'


def test_fit_with_precomputed(mipha_predictor, data_source_mock):
    data_sources = [data_source_mock]
    train_labels = [1, 0, 1]
    precomputed_features = {('data_source_1', 'extractor_1'): 'precomputed_features'}

    with patch.object(mipha_predictor, 'process_data', return_value='processed_data') as process_data_mock:
        with patch.object(mipha_predictor.model, 'fit', return_value='model_output') as model_fit_mock:
            result = mipha_predictor.fit(data_sources, train_labels, precomputed_features)

            process_data_mock.assert_called_once_with(data_sources, precomputed_features)
            model_fit_mock.assert_called_once_with('processed_data', train_labels)

            assert result == 'model_output'


def test_fit_with_additional_args_kwargs(mipha_predictor, data_source_mock):
    data_sources = [data_source_mock]
    train_labels = [1, 0, 1]

    additional_args = ('arg1', 'arg2')
    additional_kwargs = {'kwarg1': 'value1', 'kwarg2': 'value2'}

    with patch.object(mipha_predictor, 'process_data', return_value='processed_data') as process_data_mock:
        with patch.object(mipha_predictor.model, 'fit', return_value='model_output') as model_fit_mock:
            result = mipha_predictor.fit(data_sources, train_labels, None, *additional_args, **additional_kwargs)

            process_data_mock.assert_called_once_with(data_sources, None)
            model_fit_mock.assert_called_once_with('processed_data', train_labels, *additional_args,
                                                   **additional_kwargs)

            assert result == 'model_output'


def test_predict_no_precomputed(mipha_predictor, data_source_mock):
    data_sources = [data_source_mock]

    with patch.object(mipha_predictor, 'process_data', return_value='processed_data') as process_data_mock:
        with patch.object(mipha_predictor.model, 'predict', return_value='prediction_output') as model_predict_mock:
            result = mipha_predictor.predict(data_sources)

            process_data_mock.assert_called_once_with(data_sources, None)
            model_predict_mock.assert_called_once_with('processed_data')

            assert result == 'prediction_output'


def test_predict_with_precomputed(mipha_predictor, data_source_mock):
    data_sources = [data_source_mock]
    precomputed_features = {('data_source_1', 'extractor_1'): 'precomputed_features'}

    with patch.object(mipha_predictor, 'process_data', return_value='processed_data') as process_data_mock:
        with patch.object(mipha_predictor.model, 'predict', return_value='prediction_output') as model_predict_mock:
            result = mipha_predictor.predict(data_sources, precomputed_features)

            process_data_mock.assert_called_once_with(data_sources, precomputed_features)
            model_predict_mock.assert_called_once_with('processed_data')

            assert result == 'prediction_output'


def test_predict_with_additional_args_kwargs(mipha_predictor, data_source_mock):
    data_sources = [data_source_mock]

    additional_args = ('arg1', 'arg2')
    additional_kwargs = {'kwarg1': 'value1', 'kwarg2': 'value2'}

    with patch.object(mipha_predictor, 'process_data', return_value='processed_data') as process_data_mock:
        with patch.object(mipha_predictor.model, 'predict', return_value='prediction_output') as model_predict_mock:
            result = mipha_predictor.predict(data_sources, None, *additional_args, **additional_kwargs)

            process_data_mock.assert_called_once_with(data_sources, None)
            model_predict_mock.assert_called_once_with('processed_data', *additional_args, **additional_kwargs)

            assert result == 'prediction_output'


def test_evaluate_no_precomputed(mipha_predictor, data_source_mock):
    data_sources = [data_source_mock]
    test_labels = [1, 0, 1]

    with patch.object(mipha_predictor, 'process_data', return_value='processed_data') as process_data_mock:
        with patch.object(mipha_predictor.evaluator, 'evaluate_model',
                          return_value='evaluation_output') as evaluate_model_mock:
            result = mipha_predictor.evaluate(data_sources, test_labels)

            process_data_mock.assert_called_once_with(data_sources, None)
            evaluate_model_mock.assert_called_once_with(mipha_predictor.model, 'processed_data', test_labels)

            assert result == 'evaluation_output'


def test_evaluate_with_precomputed(mipha_predictor, data_source_mock):
    data_sources = [data_source_mock]
    test_labels = [1, 0, 1]
    precomputed_features = {('data_source_1', 'extractor_1'): 'precomputed_features'}

    with patch.object(mipha_predictor, 'process_data', return_value='processed_data') as process_data_mock:
        with patch.object(mipha_predictor.evaluator, 'evaluate_model',
                          return_value='evaluation_output') as evaluate_model_mock:
            result = mipha_predictor.evaluate(data_sources, test_labels, precomputed_features)

            process_data_mock.assert_called_once_with(data_sources, precomputed_features)
            evaluate_model_mock.assert_called_once_with(mipha_predictor.model, 'processed_data', test_labels)

            assert result == 'evaluation_output'


def test_evaluate_with_additional_args_kwargs(mipha_predictor, data_source_mock):
    data_sources = [data_source_mock]
    test_labels = [1, 0, 1]

    additional_args = ('arg1', 'arg2')
    additional_kwargs = {'kwarg1': 'value1', 'kwarg2': 'value2'}

    with patch.object(mipha_predictor, 'process_data', return_value='processed_data') as process_data_mock:
        with patch.object(mipha_predictor.evaluator, 'evaluate_model',
                          return_value='evaluation_output') as evaluate_model_mock:
            result = mipha_predictor.evaluate(data_sources, test_labels, None, *additional_args, **additional_kwargs)

            process_data_mock.assert_called_once_with(data_sources, None)
            evaluate_model_mock.assert_called_once_with(mipha_predictor.model, 'processed_data', test_labels,
                                                        *additional_args, **additional_kwargs)

            assert result == 'evaluation_output'
