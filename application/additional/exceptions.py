
class BadConfigurationError(Exception):
    """Exception raised in the cases of bad input configuration.

    Attributes:
        subtype -- the configuration step which caused the error
        message -- explanation of the error
    """

    def __init__(self, subtype, message="The error was caused by bad input "
                                        "configuration"):
        self.message = f'{message} in {subtype}'
        super().__init__(self.message)


class ModelNotLoadedProperlyError(Exception):
    """Exception raised in the cases of training model not loading properly.

    Attributes:
        model_name -- model_name of the model that was not loaded
        model_version -- model_version of the model that was not loaded
        message -- explanation of the error
    """

    def __init__(self, model_name, model_version, message="The error was caused by the model not loading properly (the file is faulty of it's not in database) for model "):
        self.message = f'{message}{model_name}-{model_version}'
        super().__init__(self.message)
