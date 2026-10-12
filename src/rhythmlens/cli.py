"""Minimal command-line entry point."""

import argparse

from rhythmlens import __version__


def main() -> None:
    """Display package information; research commands are pending."""
    parser = argparse.ArgumentParser(prog="rhythmlens", description="RhythmLens ECG research tools")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args()
    parser.print_help()
