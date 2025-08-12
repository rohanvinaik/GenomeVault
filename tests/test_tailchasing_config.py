"""
Tests for tailchasing configuration and calibration functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tailchasing.config import (
    CalibrationResult,
    ConfigManager,
    DistanceWeights,
    PolymerConfig,
    get_config,
    get_config_manager,
    save_config,
)


class TestDistanceWeights:
    """Test DistanceWeights configuration."""

    def test_default_weights(self):
        """Test default distance weights."""
        weights = DistanceWeights()

        assert weights.tok == 1.0
        assert weights.ast == 2.0
        assert weights.mod == 3.0
        assert weights.git == 4.0

    def test_custom_weights(self):
        """Test custom distance weights."""
        weights = DistanceWeights(tok=0.5, ast=1.5, mod=2.5, git=3.5)

        assert weights.tok == 0.5
        assert weights.ast == 1.5
        assert weights.mod == 2.5
        assert weights.git == 3.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        weights = DistanceWeights(tok=1.1, ast=2.2, mod=3.3, git=4.4)
        weight_dict = weights.to_dict()

        expected = {"tok": 1.1, "ast": 2.2, "mod": 3.3, "git": 4.4}
        assert weight_dict == expected

    def test_from_dict(self):
        """Test creation from dictionary."""
        weight_dict = {"tok": 1.5, "ast": 2.5, "mod": 3.5, "git": 4.5}
        weights = DistanceWeights.from_dict(weight_dict)

        assert weights.tok == 1.5
        assert weights.ast == 2.5
        assert weights.mod == 3.5
        assert weights.git == 4.5

    def test_from_dict_partial(self):
        """Test creation from partial dictionary."""
        weight_dict = {"tok": 1.5, "mod": 3.5}
        weights = DistanceWeights.from_dict(weight_dict)

        assert weights.tok == 1.5
        assert weights.ast == 2.0  # Default
        assert weights.mod == 3.5
        assert weights.git == 4.0  # Default


class TestPolymerConfig:
    """Test PolymerConfig configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PolymerConfig()

        assert config.alpha == 1.2
        assert isinstance(config.weights, DistanceWeights)
        assert config.epsilon == 1e-6
        assert config.kappa == 1.0
        assert config.max_distance == 1000
        assert config.min_contact_threshold == 0.01
        assert len(config.tad_patterns) > 0

    def test_custom_config(self):
        """Test custom configuration values."""
        weights = DistanceWeights(tok=2.0, ast=3.0, mod=4.0, git=5.0)
        config = PolymerConfig(
            alpha=1.3,
            weights=weights,
            epsilon=1e-5,
            kappa=2.0,
            max_distance=500,
            min_contact_threshold=0.05,
        )

        assert config.alpha == 1.3
        assert config.weights == weights
        assert config.epsilon == 1e-5
        assert config.kappa == 2.0
        assert config.max_distance == 500
        assert config.min_contact_threshold == 0.05

    def test_alpha_validation(self):
        """Test alpha parameter validation."""
        # Valid alpha values
        PolymerConfig(alpha=1.0)  # Should not raise
        PolymerConfig(alpha=1.2)  # Should not raise
        PolymerConfig(alpha=1.5)  # Should not raise

        # Invalid alpha values
        with pytest.raises(ValueError, match="alpha must be between 1.0 and 1.5"):
            PolymerConfig(alpha=0.9)

        with pytest.raises(ValueError, match="alpha must be between 1.0 and 1.5"):
            PolymerConfig(alpha=1.6)

    def test_epsilon_validation(self):
        """Test epsilon parameter validation."""
        PolymerConfig(epsilon=1e-6)  # Should not raise

        with pytest.raises(ValueError, match="epsilon must be positive"):
            PolymerConfig(epsilon=0.0)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            PolymerConfig(epsilon=-1e-6)

    def test_kappa_validation(self):
        """Test kappa parameter validation."""
        PolymerConfig(kappa=1.0)  # Should not raise

        with pytest.raises(ValueError, match="kappa must be positive"):
            PolymerConfig(kappa=0.0)

        with pytest.raises(ValueError, match="kappa must be positive"):
            PolymerConfig(kappa=-1.0)

    def test_max_distance_validation(self):
        """Test max_distance parameter validation."""
        PolymerConfig(max_distance=1000)  # Should not raise

        with pytest.raises(ValueError, match="max_distance must be positive"):
            PolymerConfig(max_distance=0)

        with pytest.raises(ValueError, match="max_distance must be positive"):
            PolymerConfig(max_distance=-100)

    def test_threshold_validation(self):
        """Test min_contact_threshold validation."""
        PolymerConfig(min_contact_threshold=0.01)  # Should not raise
        PolymerConfig(min_contact_threshold=0.0)  # Should not raise
        PolymerConfig(min_contact_threshold=1.0)  # Should not raise

        with pytest.raises(ValueError, match="min_contact_threshold must be between 0 and 1"):
            PolymerConfig(min_contact_threshold=-0.1)

        with pytest.raises(ValueError, match="min_contact_threshold must be between 0 and 1"):
            PolymerConfig(min_contact_threshold=1.1)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = PolymerConfig(alpha=1.3)
        config_dict = config.to_dict()

        assert config_dict["alpha"] == 1.3
        assert "weights" in config_dict
        assert "tad_patterns" in config_dict
        assert "epsilon" in config_dict
        assert "kappa" in config_dict
        assert "max_distance" in config_dict
        assert "min_contact_threshold" in config_dict

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            "alpha": 1.4,
            "weights": {"tok": 1.5, "ast": 2.5, "mod": 3.5, "git": 4.5},
            "tad_patterns": ["*.test.*"],
            "epsilon": 1e-5,
            "kappa": 2.0,
            "max_distance": 500,
            "min_contact_threshold": 0.02,
        }

        config = PolymerConfig.from_dict(config_dict)

        assert config.alpha == 1.4
        assert config.weights.tok == 1.5
        assert config.tad_patterns == ["*.test.*"]
        assert config.epsilon == 1e-5
        assert config.kappa == 2.0
        assert config.max_distance == 500
        assert config.min_contact_threshold == 0.02

    def test_from_dict_partial(self):
        """Test creation from partial dictionary."""
        config_dict = {"alpha": 1.3}
        config = PolymerConfig.from_dict(config_dict)

        assert config.alpha == 1.3
        # Other values should be defaults
        assert config.epsilon == 1e-6
        assert config.kappa == 1.0


class TestPolymerConfigFileOperations:
    """Test file operations for PolymerConfig."""

    def test_save_and_load_yaml(self):
        """Test saving and loading configuration from YAML."""
        config = PolymerConfig(alpha=1.3, kappa=2.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Save to file
            config.save_to_file(temp_path)
            assert temp_path.exists()

            # Verify file contents
            with open(temp_path, "r") as f:
                yaml_data = yaml.safe_load(f)

            assert yaml_data["alpha"] == 1.3
            assert yaml_data["kappa"] == 2.0

            # Load from file
            loaded_config = PolymerConfig.load_from_file(temp_path)

            assert loaded_config.alpha == 1.3
            assert loaded_config.kappa == 2.0

        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        nonexistent_path = Path("/nonexistent/config.yml")

        with pytest.raises(FileNotFoundError):
            PolymerConfig.load_from_file(nonexistent_path)

    def test_load_or_default(self):
        """Test load_or_default behavior."""
        # Non-existent file should return default
        config = PolymerConfig.load_or_default("/nonexistent/config.yml")
        assert config.alpha == 1.2  # Default value

        # Existing file should load from file
        temp_config = PolymerConfig(alpha=1.4)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            temp_config.save_to_file(temp_path)
            loaded_config = PolymerConfig.load_or_default(temp_path)

            assert loaded_config.alpha == 1.4

        finally:
            temp_path.unlink(missing_ok=True)

    def test_get_default_config_path(self):
        """Test getting default configuration path."""
        default_path = PolymerConfig.get_default_config_path()

        assert isinstance(default_path, Path)
        assert ".tailchasing_polymer.yml" in str(default_path)


class TestCalibrationResult:
    """Test CalibrationResult data structure."""

    def test_calibration_result_creation(self):
        """Test CalibrationResult creation."""
        weights = DistanceWeights(tok=1.5, ast=2.5, mod=3.5, git=4.5)
        result = CalibrationResult(
            optimal_alpha=1.3,
            optimal_weights=weights,
            correlation_score=0.85,
            validation_score=0.80,
            iterations=50,
            converged=True,
        )

        assert result.optimal_alpha == 1.3
        assert result.optimal_weights == weights
        assert result.correlation_score == 0.85
        assert result.validation_score == 0.80
        assert result.iterations == 50
        assert result.converged is True

    def test_to_dict(self):
        """Test CalibrationResult to dictionary conversion."""
        weights = DistanceWeights(tok=1.5, ast=2.5, mod=3.5, git=4.5)
        result = CalibrationResult(
            optimal_alpha=1.3,
            optimal_weights=weights,
            correlation_score=0.85,
            validation_score=0.80,
            iterations=50,
            converged=True,
        )

        result_dict = result.to_dict()

        assert result_dict["optimal_alpha"] == 1.3
        assert result_dict["correlation_score"] == 0.85
        assert result_dict["validation_score"] == 0.80
        assert result_dict["iterations"] == 50
        assert result_dict["converged"] is True
        assert "optimal_weights" in result_dict

    def test_from_dict(self):
        """Test CalibrationResult from dictionary creation."""
        result_dict = {
            "optimal_alpha": 1.3,
            "optimal_weights": {"tok": 1.5, "ast": 2.5, "mod": 3.5, "git": 4.5},
            "correlation_score": 0.85,
            "validation_score": 0.80,
            "iterations": 50,
            "converged": True,
        }

        result = CalibrationResult.from_dict(result_dict)

        assert result.optimal_alpha == 1.3
        assert result.optimal_weights.tok == 1.5
        assert result.correlation_score == 0.85
        assert result.validation_score == 0.80
        assert result.iterations == 50
        assert result.converged is True


class TestConfigManager:
    """Test ConfigManager functionality."""

    def test_config_manager_creation(self):
        """Test ConfigManager creation."""
        manager = ConfigManager()

        assert manager.config_path is None
        assert manager._config is None

    def test_config_manager_with_path(self):
        """Test ConfigManager with specific path."""
        test_path = Path("/test/config.yml")
        manager = ConfigManager(test_path)

        assert manager.config_path == test_path

    def test_load_config_default(self):
        """Test loading default configuration."""
        manager = ConfigManager()
        config = manager.load_config()

        # Should load default config since no file exists
        assert config.alpha == 1.2
        assert config.kappa == 1.0

    def test_save_config(self):
        """Test saving configuration."""
        config = PolymerConfig(alpha=1.4)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            manager = ConfigManager(temp_path)
            manager.save_config(config)

            assert temp_path.exists()

            # Load and verify
            loaded_config = PolymerConfig.load_from_file(temp_path)
            assert loaded_config.alpha == 1.4

        finally:
            temp_path.unlink(missing_ok=True)

    def test_update_from_calibration(self):
        """Test updating configuration from calibration results."""
        weights = DistanceWeights(tok=1.5, ast=2.5, mod=3.5, git=4.5)
        calibration_result = CalibrationResult(
            optimal_alpha=1.3,
            optimal_weights=weights,
            correlation_score=0.85,
            validation_score=0.80,
            iterations=50,
            converged=True,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            manager = ConfigManager(temp_path)
            manager.update_from_calibration(calibration_result)

            # Verify updated configuration
            updated_config = manager.config
            assert updated_config.alpha == 1.3
            assert updated_config.weights.tok == 1.5

        finally:
            temp_path.unlink(missing_ok=True)

    def test_validate_config(self):
        """Test configuration validation."""
        manager = ConfigManager()

        # Valid configuration
        valid_config = PolymerConfig()
        issues = manager.validate_config(valid_config)
        assert len([issue for issue in issues if not issue.startswith("Warning:")]) == 0

        # Invalid configuration
        with pytest.raises(ValueError):
            PolymerConfig(alpha=0.5)  # This should raise during __post_init__

    @patch.dict("os.environ", {"TAILCHASING_ALPHA": "1.4"})
    def test_environment_overrides(self):
        """Test environment variable overrides."""
        manager = ConfigManager()
        overrides = manager.get_environment_overrides()

        assert overrides["alpha"] == 1.4

    @patch.dict("os.environ", {"TAILCHASING_ALPHA": "invalid"})
    def test_invalid_environment_overrides(self):
        """Test invalid environment variable values."""
        manager = ConfigManager()

        with pytest.raises(ValueError, match="Invalid environment variable"):
            manager.get_environment_overrides()

    def test_apply_environment_overrides(self):
        """Test applying environment overrides to configuration."""
        manager = ConfigManager()
        config = PolymerConfig(alpha=1.2)

        overrides = {"alpha": 1.4, "kappa": 2.0}

        # Mock the get_environment_overrides method
        with patch.object(manager, "get_environment_overrides", return_value=overrides):
            updated_config = manager.apply_environment_overrides(config)

            assert updated_config.alpha == 1.4
            assert updated_config.kappa == 2.0


class TestGlobalConfigFunctions:
    """Test global configuration functions."""

    def test_get_config_manager(self):
        """Test global config manager function."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()

        # Should return the same instance
        assert manager1 is manager2

    def test_get_config_manager_with_path(self):
        """Test global config manager with specific path."""
        test_path = Path("/test/config.yml")
        manager = get_config_manager(test_path)

        assert manager.config_path == test_path

    def test_get_config(self):
        """Test global get_config function."""
        config = get_config()

        assert isinstance(config, PolymerConfig)
        assert config.alpha == 1.2  # Default value

    def test_save_config_global(self):
        """Test global save_config function."""
        config = PolymerConfig(alpha=1.3)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_config(config, temp_path)

            # Verify file was created
            assert temp_path.exists()

            # Load and verify
            loaded_config = PolymerConfig.load_from_file(temp_path)
            assert loaded_config.alpha == 1.3

        finally:
            temp_path.unlink(missing_ok=True)


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""

    def test_config_with_empty_tad_patterns(self):
        """Test configuration with empty TAD patterns."""
        config = PolymerConfig(tad_patterns=[])

        assert config.tad_patterns == []

    def test_config_with_extreme_values(self):
        """Test configuration with extreme but valid values."""
        config = PolymerConfig(
            alpha=1.0,
            epsilon=1e-15,
            kappa=0.001,
            max_distance=1,
            min_contact_threshold=0.0,
        )

        # Should not raise any validation errors
        assert config.alpha == 1.0
        assert config.epsilon == 1e-15
        assert config.kappa == 0.001
        assert config.max_distance == 1
        assert config.min_contact_threshold == 0.0

    def test_config_from_dict_with_invalid_weights(self):
        """Test configuration from dictionary with invalid weights structure."""
        # Weights as non-dict should use defaults
        config_dict = {"alpha": 1.3, "weights": "invalid"}
        config = PolymerConfig.from_dict(config_dict)

        assert config.alpha == 1.3
        assert config.weights.tok == 1.0  # Default

    def test_config_serialization_roundtrip(self):
        """Test configuration serialization round-trip."""
        original_config = PolymerConfig(
            alpha=1.35,
            weights=DistanceWeights(tok=1.1, ast=2.2, mod=3.3, git=4.4),
            tad_patterns=["*.custom.*"],
            epsilon=1e-5,
            kappa=1.5,
        )

        # Convert to dict and back
        config_dict = original_config.to_dict()
        restored_config = PolymerConfig.from_dict(config_dict)

        assert restored_config.alpha == original_config.alpha
        assert restored_config.weights.tok == original_config.weights.tok
        assert restored_config.weights.ast == original_config.weights.ast
        assert restored_config.weights.mod == original_config.weights.mod
        assert restored_config.weights.git == original_config.weights.git
        assert restored_config.tad_patterns == original_config.tad_patterns
        assert restored_config.epsilon == original_config.epsilon
        assert restored_config.kappa == original_config.kappa


if __name__ == "__main__":
    pytest.main([__file__])
