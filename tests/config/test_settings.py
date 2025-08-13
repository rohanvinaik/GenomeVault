from genomevault.config.settings import settings


def test_settings_defaults():
    """Test settings defaults.
    Returns:
        Result of the operation."""
    assert settings.api_port == 8000
    assert settings.log_level in ("INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL")
