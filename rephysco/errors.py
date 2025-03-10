"""
Error classes for the Rephysco LLM System.
"""


class RephyscoError(Exception):
    """Base class for all Rephysco errors."""
    pass


class ProviderError(RephyscoError):
    """Error raised when a provider encounters an error."""
    pass


class AuthenticationError(ProviderError):
    """Error raised when authentication fails."""
    pass


class RateLimitError(ProviderError):
    """Error raised when a rate limit is exceeded."""
    pass


class ContentFilterError(ProviderError):
    """Error raised when content is filtered by the provider."""
    pass


class InvalidRequestError(ProviderError):
    """Error raised when a request is invalid."""
    pass


class ServiceUnavailableError(ProviderError):
    """Error raised when a service is unavailable."""
    pass 