#!/usr/bin/env python
"""
Example script for running an agent session directly.

This script demonstrates how to use the agent session implementation
without going through the CLI.
"""

import argparse
from rephysco.types import ModelProvider
from rephysco.examples.agent_session import run_agent_session

def main():
    """Run the agent session with command-line arguments."""
    parser = argparse.ArgumentParser(description="Run an agent session with Google Search capability")
    
    parser.add_argument(
        "--provider", "-p",
        choices=[p.value for p in ModelProvider],
        default=ModelProvider.OPENAI.value,
        help="LLM provider to use"
    )
    
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Specific model to use (if not specified, uses provider default)"
    )
    
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Temperature for generation (0.0 to 1.0)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        run_agent_session(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("\nAgent session ended")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 