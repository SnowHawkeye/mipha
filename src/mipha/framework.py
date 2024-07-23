import json
import os
import tempfile
import uuid
import zipfile
from abc import ABC, abstractmethod
import _pickle as pickle
from dataclasses import dataclass
from typing import Self, Any, Dict, Tuple


@dataclass
class DataSource:
    """
    Represents a data source with a specific type, name, and data.

    Attributes:
        data_type (str): The type of data contained in the data source.
            This is not a python type, but rather a user-defined convention to signal which data sources
            use the same kind of data (and can be handled by the same type of feature extractors).
        name (str): The name of the data source.
        data (Any): The actual data of the data source, which can be of any type.
    """

    data_type: str
    name: str
    data: Any


class MiphaComponent(ABC):
    """
    Base class for Mipha components.
    """

    def __init__(self, component_name: str = None):
        if component_name is None:
            component_name = self.__class__.__name__ + "_" + str(uuid.uuid4())
        self.component_name = component_name

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

    def __init__(self, component_name: str = None, managed_data_types: list[str] = None):
        """
        :param managed_data_types: A list of data source type names to extract features from.
        Whenever feature extraction needs to be performed for a DataSource with attribute dataType
        equal to managed_data_types, the MiphaPredictor: will feed it to this FeatureExtractor.
        """
        super().__init__(component_name)
        self.managed_data_types = managed_data_types

    @abstractmethod
    def extract_features(self, x):
        """
        :param x: Input data whose features should be extracted.
        :return: Extracted features. Must be able to cast to a numpy array.
        """
        return NotImplemented


class DataContract:
    def __init__(self, mappings: dict[str, FeatureExtractor | list[FeatureExtractor]] = None):
        """
        Initializes the DataContract with optional initial mappings.

        :param mappings: Optional dictionary where keys are string names of data source types and values are
                                 single or multiple FeatureExtractor instances.
        """
        self.contract: dict[str, list[FeatureExtractor]] = {}
        if mappings:
            self.add_data_sources(mappings)

    def add_data_sources(self, mappings: dict[str, FeatureExtractor | list[FeatureExtractor]]) -> Self:
        """
        Adds mappings of data source type names to FeatureExtractor instances to the contract.

        :param mappings: A dictionary where keys are string names of data sources and values are
                         single or multiple FeatureExtractor instances.
        """
        if not isinstance(mappings, dict):
            raise ValueError("Mappings must be a dictionary")
        for data_source_type, extractors in mappings.items():
            if not isinstance(extractors, list):
                extractors = [extractors]
            if data_source_type in self.contract:
                print(f"Warning: Extractors for {data_source_type} are already in the contract and will be replaced.")
            self.contract[data_source_type] = extractors

        return self

    def get_extractors(self, data_source_type: str) -> list[FeatureExtractor]:
        """
        Retrieves the list of FeatureExtractor instances for a given data source type name.

        :param data_source_type: A string name of a data source type.
        :return: A list of FeatureExtractor instances associated with the data_source_type.
        :raises KeyError: If the data_source_type is not present in the contract.
        """
        if data_source_type not in self.contract:
            raise KeyError(f"{data_source_type} data source type is not in the contract")
        return self.contract[data_source_type]

    @staticmethod
    def from_feature_extractors(extractors: list[FeatureExtractor]) -> 'DataContract':
        """
        Creates a DataContract from a list of FeatureExtractor instances by mapping each extractor to its managed data sources.

        :param extractors: A list of FeatureExtractor instances.
        :return: An instance of DataContract initialized with the mappings from the feature extractors.
        """
        mappings = {}
        for extractor in extractors:
            managed_data_types = extractor.managed_data_types
            if managed_data_types:
                for data_source_type in managed_data_types:
                    if data_source_type not in mappings:
                        mappings[data_source_type] = []
                    mappings[data_source_type].append(extractor)

        return DataContract(mappings=mappings)


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
        self.data_contract = DataContract.from_feature_extractors(self.feature_extractors)
        self.last_computed_features = None

    def process_data(self, data_sources: list[DataSource],
                     precomputed_features: Dict[Tuple[str, str], Any] = None):
        """
        Utility function to process the data sources by extracting and aggregating features.
        :param data_sources: A list of data sources to process.
        :param precomputed_features: A mapping of (data source name, feature extractor name) tuples
        to already extracted features.
        Data processing will be skipped for the provided mappings, and the given extraction will be used instead.
        :return: The result of the aggregation.
        """
        print("Extracting features from data sources...")

        extracted_features = []
        computed_features = {} if precomputed_features is None else precomputed_features

        for data_source in data_sources:
            extractors = self.data_contract.get_extractors(data_source.data_type)
            for extractor in extractors:
                # attempt to retrieve result from mapping
                features = computed_features.get((data_source.name, extractor.component_name))
                if features is None:
                    features = extractor.extract_features(data_source.data)
                else:
                    print(f"Using precomputed features by {extractor.component_name} for {data_source.name}")
                extracted_features.append(features)
                computed_features[(data_source.name, extractor.component_name)] = features

        self.last_computed_features = computed_features
        print("Feature extraction complete!")
        print("The last feature extraction can be accessed by using MiphaPredictor.last_computed_features.\n")

        print("Aggregating features from data sources...")
        aggregation = self.aggregator.aggregate_features(extracted_features)
        print("Aggregation complete!\n")

        return aggregation

    def fit(self, data_sources: list[DataSource], train_labels,
            precomputed_features: Dict[Tuple[str, str], Any] = None, *args, **kwargs):
        """Fit the MIPHA model to the given data sources ("x_train") using the provided training labels ("y_train")."""
        print("Fitting the model...")
        x_train = self.process_data(data_sources, precomputed_features)

        output = self.model.fit(x_train, train_labels, *args, **kwargs)
        print("Model fit successfully!\n")

        return output

    def predict(self, data_sources: list[DataSource],
                precomputed_features: Dict[Tuple[str, str], Any] = None, *args, **kwargs):
        """Use the MIPHA model to predict a label for the given data sources ("x_test")."""
        x_test = self.process_data(data_sources, precomputed_features)
        return self.model.predict(x_test, *args, **kwargs)

    def evaluate(self, data_sources: list[DataSource], test_labels,
                 precomputed_features: Dict[Tuple[str, str], Any] = None, *args, **kwargs):
        """Evaluate the MIPHA model on the given data sources ("x_test"),
        using the provided test labels ("y_test") as reference."""
        x_test = self.process_data(data_sources, precomputed_features)
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
