
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
