import json
import os
import tempfile
import zipfile
from abc import ABC, abstractmethod
import _pickle as pickle


class MiphaComponent(ABC):
    """
    Base class for Mipha components.
    """

    def save(self, filename):
        """
        Save component to file.

        Parameters:
        :param filename: The name of the file where the object will be saved.
        """

        # Ensure the file has a .pkl extension
        if not filename.lower().endswith('.pkl'):
            filename += '.pkl'
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
            print(f"Component {type(self).__name__} successfully saved to {filename}")
        except Exception as e:
            print(f"An error occurred while saving the object: {e}")

    @staticmethod
    def load(path):
        try:
            with open(path, 'rb') as file:
                return pickle.load(file)  # Example placeholder
        except Exception as e:
            print(f"An error occurred while loading the object: {e}")


class FeatureExtractor(MiphaComponent):
    """
    Feature Extractor base class. The purpose of a Feature Extractor is to output features from a data source.
    """

    @abstractmethod
    def extract_features(self, x):
        """
        :param x: Input data whose features should be extracted.
        :return: Extracted features. Must be able to cast to a numpy array.
        """
        return NotImplemented


class Aggregator(MiphaComponent):
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
        return NotImplemented


class MachineLearningModel(MiphaComponent):
    """
    Wrapper for machine learning models used within the MIPHA framework.
    """

    @abstractmethod
    def fit(self, x_train, y_train, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def predict(self, x_test, *args, **kwargs):
        return NotImplemented


class Evaluator(MiphaComponent):
    """
    Evaluator base class. Its implementations can customize how the model's performance is evaluated.
    """

    @abstractmethod
    def evaluate_model(self, model: MachineLearningModel, x_test, y_test, *args, **kwargs):
        return NotImplemented


class MiphaPredictor:
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

    def save(self, archive_path):
        """
        Save the MIPHA model and its components into a ZIP archive.

        :param archive_path: Path to the ZIP archive where the model and components will be saved.
        """
        # Ensure the file has a .zip extension
        if not archive_path.lower().endswith('.zip'):
            archive_path += '.zip'

        # Create a temporary directory to store the individual files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Metadata dictionary to store component types
            metadata = {}

            # Save each feature extractor individually
            for idx, component in enumerate(self.feature_extractors):
                component_type = type(component).__name__
                component_path = os.path.join(temp_dir, f"feature_extractor_{idx}.pkl")
                component.save(component_path)
                metadata[f"feature_extractor_{idx}"] = component_type
                print(f"Component feature extractor {idx} ({component_type}) saved to {component_path}")

            # Save other components
            components = {
                'aggregator': self.aggregator,
                'model': self.model,
                'evaluator': self.evaluator
            }
            for name, component in components.items():
                component_type = type(component).__name__
                component_path = os.path.join(temp_dir, f"{name}.pkl")
                component.save(component_path)
                metadata[name] = component_type
                print(f"Component '{name}' ({component_type}) saved to {component_path}")

            # Save metadata to a JSON file
            metadata_path = os.path.join(temp_dir, "metadata.json")
            with open(metadata_path, 'w') as metadata_file:
                json.dump(metadata, metadata_file)

            # Create a ZIP archive from the temporary directory
            with zipfile.ZipFile(archive_path, 'w') as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, temp_dir))
            print(f"Model saved to ZIP archive {archive_path}")

    @classmethod
    def load(cls, archive_path):
        """
        Load the MIPHA model from a ZIP archive.
        This method only works for default implementations of MiphaComponent.load().
        If subcomponents define custom load() functions, they must be loaded individually.

        :param archive_path: Path to the ZIP archive where the model and components are saved.
        :return: An instance of MIPHA.
        """
        # Extract the ZIP archive to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(temp_dir)

            # Load metadata
            metadata_path = os.path.join(temp_dir, "metadata.json")
            with open(metadata_path, 'r') as metadata_file:
                metadata = json.load(metadata_file)

            # Load feature extractors
            feature_extractors = []
            idx = 0
            while f"feature_extractor_{idx}" in metadata:
                # Retrieve the class from metadata. However globals() does not seem to have the class types
                # This means we will only call MiphaComponent.load for now
                # See the following issue: https://github.com/SnowHawkeye/mipha/issues/24
                # component_type = metadata[f"feature_extractor_{idx}"]
                # component_class = globals()[component_type]

                component_path = os.path.join(temp_dir, f"feature_extractor_{idx}.pkl")
                component = MiphaComponent.load(component_path)
                feature_extractors.append(component)
                idx += 1

            # Load other components
            components = {}
            for name in ['aggregator', 'model', 'evaluator']:
                if name in metadata:
                    # See comment above
                    # component_type = metadata[name]
                    # component_class = globals()[component_type]

                    component_path = os.path.join(temp_dir, f"{name}.pkl")
                    components[name] = MiphaComponent.load(component_path)

            return cls(
                feature_extractors=feature_extractors,
                aggregator=components.get('aggregator'),
                model=components.get('model'),
                evaluator=components.get('evaluator')
            )
