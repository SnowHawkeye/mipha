import _pickle as pickle
import os
import tempfile

from src.mipha.framework import MiphaComponent, MiphaPredictor


class MockComponent(MiphaComponent):
    def __init__(self, name):
        self.name = name


# noinspection PyTypeChecker
def test_save_and_load():
    # Create mock components
    feature_extractors = [MockComponent(name=f"feature_{i}") for i in range(3)]
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
