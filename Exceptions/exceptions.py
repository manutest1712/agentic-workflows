class MissingUserStoriesError(RuntimeError):
    """Raised when a step is executed without user stories."""
    pass

class MissingFeaturesError(Exception):
    """Raised when product features are required but not found in workflow_context."""
    pass

class MissingTasksError(Exception):
    """Raised when engineering tasks are required but not found in workflow_context."""
    pass