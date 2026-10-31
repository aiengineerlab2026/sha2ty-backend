class DomainError(Exception):
    """A business rule was broken; the API maps it to a 4xx."""
