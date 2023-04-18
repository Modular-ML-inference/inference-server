class BadDataFormatException(Exception):
    """Exception raised when the format of the data does not comply with the selected schema.

    Attributes:
        format -- the non-compliant format
        schema -- the schema of the chosen format
        schema_id -- the id defining said schema"""

    def __init__(self, format, schema, schema_id):
        self.format = format
        self.schema = schema
        self.schema_id = schema_id
        self.message = f"The current data format is non-compliant with the schema type of {schema_id}"
        super().__init__(self.message)

class TransformationConfigurationInvalidException(Exception):
    """Exception raised when the supported configuration of transformation is invalid 

    Attributes:
        transformation_id -- the id of the specific transformation that triggers the error"""

    def __init__(self,  transformation_id):
        self.transformation_id = transformation_id
        self.message = f"The transformation " \
                       f"{self.transformation_id} could not be properly loaded"
        super().__init__(self.message)

class TransformationPipelineConfigurationInsufficientException(Exception):
    """Exception raised when the format and pipeline configuration is not sufficient in order 
    to transform the data into a model-compliant form. """

    def __init__(self):
        self.message = "The format and pipeline configuration is not sufficient in order to transform the data into a model-compliant form."
        super().__init__(self.message)

class DataTransformationModelUnavailableException(Exception):
    """Exception raised when the data transformation relies on the existence
    in the database of a model of a given name and version.

    Attributes:
        model_name -- the name of the necessary model
        model_version -- the version that the data transformation seeks
        transformation_id -- the id of the specific transformation that needs the model"""

    def __init__(self, model_name, model_version, transformation_id):
        self.model_name = model_name
        self.model_version = model_version
        self.transformation_id = transformation_id
        self.message = f"The model {self.model_name} v.{self.model_version} requested by transformation " \
                       f"{self.transformation_id} could not be loaded"
        super().__init__(self.message)


class NotEnoughResourcesException(Exception):
    """An exception thrown in the case of insufficient resources to use a functionality.

    Attributes:
        reason -- what was the insufficient resource
        value_expected -- what is the value of the resource that was expected
        value_available -- what is the value of the resource that is available"""

    def __init__(self, resource, value_expected, value_available):
        self.resource = resource
        self.value_expected = value_expected
        self.value_available = value_available
        self.message = f"The amount of available {self.resource} is {self.value_available} " \
                       f"when it is expected to be {self.value_expected}"
        super().__init__(self.message)


class NonCompliantPackagesException(Exception):
    """An exception thrown in the case of the necessary package list not being fulfilled for the functionality.

    Attributes:
        package_name -- what is the name of the expected package
        version_expected -- what is the version of the package that should be delivered
        version_available -- what is the available version of this package.
        May also default to "None" in the absence of any package"""

    def __init__(self, package_name, version_expected, version_available):
        self.package_name = package_name
        self.version_expected = version_expected
        self.version_available = version_available
        self.message = f"The package {self.package_name} is expected to be available in {self.version_expected} " \
                       f"when it is actually available as {self.version_available}"
        super().__init__(self.message)
