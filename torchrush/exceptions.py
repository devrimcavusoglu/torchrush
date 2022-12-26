class DatasetNotLoadedError(LookupError):
    def __init__(self, message: str = None):
        message = (
            message or "Dataset is not yet loaded, to operate with dataset call `load()` method first."
        )
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
