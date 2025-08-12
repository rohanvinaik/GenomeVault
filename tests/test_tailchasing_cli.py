"""
Tests for tailchasing CLI functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from tailchasing.cli import TailchasingCLI, main
from tailchasing.config import PolymerConfig


class TestTailchasingCLI:
    """Test TailchasingCLI functionality."""

    def test_cli_creation(self):
        """Test CLI creation."""
        cli = TailchasingCLI()

        assert cli.console is not None

    def test_create_parser(self):
        """Test argument parser creation."""
        cli = TailchasingCLI()
        parser = cli.create_parser()

        assert parser is not None
        assert parser.prog == "tailchasing"

    def test_parser_config_subcommands(self):
        """Test config subcommand parsing."""
        cli = TailchasingCLI()
        parser = cli.create_parser()

        # Test config show
        args = parser.parse_args(["config", "show"])
        assert args.command == "config"
        assert args.config_action == "show"

        # Test config validate
        args = parser.parse_args(["config", "validate"])
        assert args.command == "config"
        assert args.config_action == "validate"

        # Test config init
        args = parser.parse_args(["config", "init"])
        assert args.command == "config"
        assert args.config_action == "init"

        # Test config init with force
        args = parser.parse_args(["config", "init", "--force"])
        assert args.command == "config"
        assert args.config_action == "init"
        assert args.force is True

    def test_parser_calibrate_command(self):
        """Test calibrate command parsing."""
        cli = TailchasingCLI()
        parser = cli.create_parser()

        # Test basic calibrate
        args = parser.parse_args(["calibrate", "--demo"])
        assert args.command == "calibrate"
        assert args.demo is True

        # Test calibrate with parameters
        args = parser.parse_args(
            [
                "calibrate",
                "--thrash-events",
                "events.json",
                "--codebase-metrics",
                "metrics.json",
                "--alpha-range",
                "1.0",
                "1.4",
                "--alpha-steps",
                "20",
                "--max-iterations",
                "200",
                "--output",
                "results.json",
            ]
        )

        assert args.thrash_events == "events.json"
        assert args.codebase_metrics == "metrics.json"
        assert args.alpha_range == [1.0, 1.4]
        assert args.alpha_steps == 20
        assert args.max_iterations == 200
        assert args.output == "results.json"

    def test_parser_validate_command(self):
        """Test validate command parsing."""
        cli = TailchasingCLI()
        parser = cli.create_parser()

        args = parser.parse_args(
            [
                "validate",
                "--test-events",
                "test_events.json",
                "--codebase-metrics",
                "metrics.json",
                "--polymer-config",
                "custom_config.yml",
            ]
        )

        assert args.command == "validate"
        assert args.test_events == "test_events.json"
        assert args.codebase_metrics == "metrics.json"
        assert args.polymer_config == "custom_config.yml"

    def test_parser_analyze_command(self):
        """Test analyze command parsing."""
        cli = TailchasingCLI()
        parser = cli.create_parser()

        args = parser.parse_args(
            [
                "analyze",
                "--input",
                "performance_data.json",
                "--output",
                "analysis_results.json",
                "--format",
                "yaml",
            ]
        )

        assert args.command == "analyze"
        assert args.input == "performance_data.json"
        assert args.output == "analysis_results.json"
        assert args.format == "yaml"

    def test_run_no_command(self):
        """Test running CLI with no command."""
        cli = TailchasingCLI()

        with patch("builtins.print") as mock_print:
            result = cli.run([])

            assert result == 1  # Error exit code

    def test_run_with_verbose(self):
        """Test running CLI with verbose flag."""
        cli = TailchasingCLI()

        with patch.object(cli, "_execute_command", return_value=0) as mock_execute:
            result = cli.run(["--verbose", "config", "show"])

            assert result == 0
            mock_execute.assert_called_once()

    def test_run_with_exception(self):
        """Test running CLI when exception occurs."""
        cli = TailchasingCLI()

        with patch.object(cli, "_execute_command", side_effect=Exception("Test error")):
            result = cli.run(["config", "show"])

            assert result == 1  # Error exit code

    def test_show_config(self):
        """Test show config command."""
        cli = TailchasingCLI()

        with patch("tailchasing.config.get_config_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.config = PolymerConfig(alpha=1.3)
            mock_get_manager.return_value = mock_manager

            result = cli._show_config(mock_manager)

            assert result == 0

    def test_show_config_error(self):
        """Test show config command with error."""
        cli = TailchasingCLI()

        mock_manager = MagicMock()
        # Properly mock the config property to raise an exception when accessed
        type(mock_manager).config = PropertyMock(side_effect=Exception("Config error"))

        result = cli._show_config(mock_manager)

        assert result == 1

    def test_validate_config_valid(self):
        """Test validate config command with valid config."""
        cli = TailchasingCLI()

        mock_manager = MagicMock()
        mock_manager.config = PolymerConfig()
        mock_manager.validate_config.return_value = []  # No issues

        result = cli._validate_config(mock_manager)

        assert result == 0

    def test_validate_config_with_warnings(self):
        """Test validate config command with warnings."""
        cli = TailchasingCLI()

        mock_manager = MagicMock()
        mock_manager.config = PolymerConfig()
        mock_manager.validate_config.return_value = ["Warning: minor issue"]

        result = cli._validate_config(mock_manager)

        assert result == 0  # Warnings don't cause error exit

    def test_validate_config_with_errors(self):
        """Test validate config command with errors."""
        cli = TailchasingCLI()

        mock_manager = MagicMock()
        mock_manager.config = PolymerConfig()
        mock_manager.validate_config.return_value = ["Error: serious problem"]

        result = cli._validate_config(mock_manager)

        assert result == 1  # Errors cause error exit

    def test_init_config_new(self):
        """Test init config command with new file."""
        cli = TailchasingCLI()

        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            temp_path = Path(temp_file.name)

        # File doesn't exist now
        assert not temp_path.exists()

        mock_manager = MagicMock()
        mock_manager.config_path = temp_path
        mock_manager.save_config = MagicMock()

        result = cli._init_config(mock_manager, force=False)

        assert result == 0
        mock_manager.save_config.assert_called_once()

    def test_init_config_exists_no_force(self):
        """Test init config command with existing file, no force."""
        cli = TailchasingCLI()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            mock_manager = MagicMock()
            mock_manager.config_path = temp_path

            result = cli._init_config(mock_manager, force=False)

            assert result == 1  # Should fail without force

        finally:
            temp_path.unlink(missing_ok=True)

    def test_init_config_exists_with_force(self):
        """Test init config command with existing file, with force."""
        cli = TailchasingCLI()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            mock_manager = MagicMock()
            mock_manager.config_path = temp_path
            mock_manager.save_config = MagicMock()

            result = cli._init_config(mock_manager, force=True)

            assert result == 0
            mock_manager.save_config.assert_called_once()

        finally:
            temp_path.unlink(missing_ok=True)

    @patch("tailchasing.cli.create_sample_thrash_events")
    @patch("tailchasing.cli.create_sample_codebase_metrics")
    @patch("tailchasing.cli.CalibrationTool")
    @patch("tailchasing.cli.get_config_manager")
    def test_calibrate_demo_success(
        self,
        mock_get_manager,
        mock_cal_tool_class,
        mock_create_codebase,
        mock_create_events,
    ):
        """Test successful calibration demo."""
        cli = TailchasingCLI()

        # Mock the dependencies
        mock_events = [MagicMock()]
        mock_codebase = MagicMock()

        # Create a proper mock result with all required attributes
        mock_result = MagicMock()
        mock_result.optimal_alpha = 1.3
        mock_result.optimal_weights = MagicMock()
        mock_result.optimal_weights.tok = 1.5
        mock_result.optimal_weights.ast = 2.5
        mock_result.optimal_weights.mod = 3.5
        mock_result.optimal_weights.git = 4.5
        mock_result.correlation_score = 0.85
        mock_result.validation_score = 0.80
        mock_result.iterations = 50
        mock_result.converged = True

        mock_create_events.return_value = mock_events
        mock_create_codebase.return_value = mock_codebase

        mock_cal_tool = MagicMock()
        mock_cal_tool.fit_parameters.return_value = mock_result
        mock_cal_tool_class.return_value = mock_cal_tool

        mock_manager = MagicMock()
        mock_manager.config = PolymerConfig()
        mock_get_manager.return_value = mock_manager

        # Mock args
        mock_args = MagicMock()
        mock_args.alpha_range = [1.0, 1.5]
        mock_args.alpha_steps = 26
        mock_args.max_iterations = 100
        mock_args.output = None
        mock_args.config = None

        # Mock user input to decline config update
        with patch("builtins.input", return_value="n"):
            result = cli._calibrate_demo(mock_args)

        assert result == 0
        mock_cal_tool.fit_parameters.assert_called_once()

    @patch("tailchasing.cli.create_sample_thrash_events")
    @patch("tailchasing.cli.create_sample_codebase_metrics")
    @patch("tailchasing.cli.CalibrationTool")
    def test_calibrate_demo_error(
        self, mock_cal_tool_class, mock_create_codebase, mock_create_events
    ):
        """Test calibration demo with error."""
        cli = TailchasingCLI()

        # Mock calibration tool to raise exception
        mock_cal_tool = MagicMock()
        mock_cal_tool.fit_parameters.side_effect = Exception("Calibration failed")
        mock_cal_tool_class.return_value = mock_cal_tool

        mock_create_events.return_value = [MagicMock()]
        mock_create_codebase.return_value = MagicMock()

        mock_args = MagicMock()
        mock_args.alpha_range = [1.0, 1.5]
        mock_args.alpha_steps = 26
        mock_args.max_iterations = 100
        mock_args.output = None
        mock_args.config = None
        mock_args.verbose = False

        result = cli._calibrate_demo(mock_args)

        assert result == 1  # Error exit code

    def test_handle_config_command(self):
        """Test config command handling."""
        cli = TailchasingCLI()

        # Test show action
        mock_args = MagicMock()
        mock_args.config = None
        mock_args.config_action = "show"

        with patch.object(cli, "_show_config", return_value=0) as mock_show:
            result = cli._handle_config_command(mock_args)

            assert result == 0
            mock_show.assert_called_once()

        # Test validate action
        mock_args.config_action = "validate"

        with patch.object(cli, "_validate_config", return_value=0) as mock_validate:
            result = cli._handle_config_command(mock_args)

            assert result == 0
            mock_validate.assert_called_once()

        # Test init action
        mock_args.config_action = "init"
        mock_args.force = False

        with patch.object(cli, "_init_config", return_value=0) as mock_init:
            result = cli._handle_config_command(mock_args)

            assert result == 0
            mock_init.assert_called_once()

    def test_handle_config_command_no_action(self):
        """Test config command with no action specified."""
        cli = TailchasingCLI()

        mock_args = MagicMock()
        mock_args.config = None
        mock_args.config_action = None

        result = cli._handle_config_command(mock_args)

        assert result == 1

    def test_handle_calibrate_command_demo(self):
        """Test calibrate command with demo flag."""
        cli = TailchasingCLI()

        mock_args = MagicMock()
        mock_args.demo = True

        with patch.object(cli, "_calibrate_demo", return_value=0) as mock_demo:
            result = cli._handle_calibrate_command(mock_args)

            assert result == 0
            mock_demo.assert_called_once()

    def test_handle_calibrate_command_missing_events(self):
        """Test calibrate command without required events file."""
        cli = TailchasingCLI()

        mock_args = MagicMock()
        mock_args.demo = False
        mock_args.thrash_events = None

        result = cli._handle_calibrate_command(mock_args)

        assert result == 1

    def test_handle_calibrate_command_missing_codebase(self):
        """Test calibrate command without required codebase file."""
        cli = TailchasingCLI()

        mock_args = MagicMock()
        mock_args.demo = False
        mock_args.thrash_events = "events.json"
        mock_args.codebase_metrics = None

        result = cli._handle_calibrate_command(mock_args)

        assert result == 1

    def test_handle_validate_command(self):
        """Test validate command handling."""
        cli = TailchasingCLI()

        mock_args = MagicMock()

        result = cli._handle_validate_command(mock_args)

        assert result == 0  # Not yet implemented, returns success

    def test_handle_analyze_command(self):
        """Test analyze command handling."""
        cli = TailchasingCLI()

        mock_args = MagicMock()

        result = cli._handle_analyze_command(mock_args)

        assert result == 0  # Not yet implemented, returns success


class TestMainFunction:
    """Test main CLI function."""

    @patch("tailchasing.cli.TailchasingCLI")
    def test_main_function(self, mock_cli_class):
        """Test main function."""
        mock_cli = MagicMock()
        mock_cli.run.return_value = 0
        mock_cli_class.return_value = mock_cli

        result = main(["config", "show"])

        assert result == 0
        mock_cli.run.assert_called_once_with(["config", "show"])

    @patch("tailchasing.cli.TailchasingCLI")
    def test_main_function_no_args(self, mock_cli_class):
        """Test main function with no arguments."""
        mock_cli = MagicMock()
        mock_cli.run.return_value = 1
        mock_cli_class.return_value = mock_cli

        result = main()

        assert result == 1
        mock_cli.run.assert_called_once_with(None)


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_cli_help(self):
        """Test CLI help output."""
        cli = TailchasingCLI()

        with pytest.raises(SystemExit) as exc_info:
            cli.run(["--help"])

        # Help should exit with code 0
        assert exc_info.value.code == 0

    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        cli = TailchasingCLI()

        # argparse will exit with SystemExit for invalid commands
        with pytest.raises(SystemExit):
            cli.run(["invalid-command"])

    def test_cli_config_show_integration(self):
        """Integration test for config show command."""
        cli = TailchasingCLI()

        # Should work with default config
        result = cli.run(["config", "show"])

        assert result == 0

    def test_cli_config_validate_integration(self):
        """Integration test for config validate command."""
        cli = TailchasingCLI()

        # Should work with default config
        result = cli.run(["config", "validate"])

        assert result == 0


if __name__ == "__main__":
    pytest.main([__file__])
