"""
Main entry point for Sparky OpenClaw agent.

Usage:
    python main.py

Requirements:
    - OpenClaw gateway should be running
    - reachy mini daemon should be running
    - Reachy robot should be connected (or run in mock mode)
"""

import asyncio
import os
import sys
import warnings
import io
import contextlib
import logging
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure logging with the required format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class StderrFilter:
    """
    Filter stderr output to suppress noisy async cleanup errors.

    Filters out "Exception ignored" messages from httpx/openai async generator cleanup.
    These are benign cleanup warnings that don't affect functionality.
    """
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.buffer = ""
        self.suppressing = False  # Track if we're currently suppressing a traceback

    def write(self, text):
        # Buffer the text
        self.buffer += text

        # If we have a complete line (ends with newline)
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            # Keep the last incomplete line in buffer
            self.buffer = lines[-1]

            # Process complete lines
            for line in lines[:-1]:
                # Check if this starts a traceback we want to suppress
                if "Exception ignored" in line:
                    self.suppressing = True
                    continue

                # If we're suppressing, check if this line is part of the traceback
                if self.suppressing:
                    # Traceback lines to suppress
                    if (
                        line.strip().startswith("Traceback") or
                        line.strip().startswith("File ") or
                        "AsyncLibraryNotFoundError" in line or
                        "RuntimeError" in line or
                        "async generator" in line or
                        "httpx" in line or
                        "openai" in line or
                        "sniffio" in line or
                        "httpcore" in line or
                        line.strip() == ""  # Empty lines in traceback
                    ):
                        # Continue suppressing
                        continue
                    else:
                        # End of traceback, stop suppressing
                        self.suppressing = False

                # Output non-suppressed lines
                if not self.suppressing:
                    self.original_stderr.write(line + '\n')

    def flush(self):
        # Flush any buffered content if not suppressing
        if self.buffer and not self.suppressing:
            self.original_stderr.write(self.buffer)
            self.buffer = ""
        self.original_stderr.flush()

    def fileno(self):
        return self.original_stderr.fileno()


# Install stderr filter
sys.stderr = StderrFilter(sys.__stderr__)

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sparky_mvp.core.state_machine import ReachyStateMachine


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create config.yaml in the current directory."
        )

    with open(config_file) as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid config file: {e}")


def check_environment() -> None:
    """
    Check required environment variables and dependencies.

    Raises:
        SystemExit: If critical requirements are missing
    """
    # Check model files
    required_models = ["models/silero_vad_v5.onnx"]
    missing_models = [p for p in required_models if not Path(p).exists()]
    if missing_models:
        print("❌ Error: Required model files not found:")
        for model in missing_models:
            print(f"  - {model}")
        print()
        sys.exit(1)

    # Wake-word model check: read the configured path from config.yaml
    try:
        config = load_config()
        ww_cfg = config.get("wake_word", {})
        ww_enabled = ww_cfg.get("enabled", False)
        ww_model = ww_cfg.get("model_path", "") or ww_cfg.get("model", "")
        if ww_enabled and ww_model and not Path(ww_model).exists():
            print(f"⚠️  Warning: Wake word model not found: {ww_model} (wake word will be disabled)")
    except Exception:
        pass  # Config load errors are handled later during startup

    print("✅ Environment check passed")
    print()


def print_banner() -> None:
    """Print startup banner."""
    print()
    print("=" * 60)
    print("  Sparky, an Alive and Useful Robot")
    print("=" * 60)
    print()
    print("  🤖 Interactive voice assistant for Reachy Mini")
    print("  🎤 Wake word: 'Wake up, Sparky'")
    print("  🎤 To chat: 'Hey Sparky'")
    print("  🎤 To sleep: 'Go to sleep, Sparky'")
    print("  ⚡ Powered by NVIDIA, Pollen Robotics, and OpenClaw")
    print()
    print("=" * 60)
    print()


def print_instructions() -> None:
    """Print usage instructions."""
    print()
    print("📋 Instructions:")
    print("  1. Say 'Wake up, Sparky' to wake the robot")
    print("  2. Initiate conversations with 'Hey Sparky'")
    print("  3. Wait for the response")
    print("  4. Overtalk loudly to interrupt")
    print("  5. Say 'Go to sleep, Sparky' to put robot to sleep")
    print("  6. Press Ctrl+C twice to exit")
    print()
    print("=" * 60)
    print()


async def main() -> None:
    """
    Main entry point.

    Loads configuration, checks environment, and starts the state machine.
    """
    try:
        # Print banner
        print_banner()

        # Check environment
        check_environment()

        # Load configuration
        print("📄 Loading configuration...")
        config = load_config()
        print("  ✓ Configuration loaded")
        print()

        # Print instructions
        print_instructions()

        # Create and run state machine
        state_machine = ReachyStateMachine(config)
        await state_machine.run()

    except KeyboardInterrupt:
        print()
        print("  👋 Goodbye!")

    except Exception as e:
        print()
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
