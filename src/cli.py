#!/usr/bin/env python3
"""
Command Line Interface for Active Inference World Simulator

Usage:
    python -m src.cli run-tiger          # Run tiger life story
    python -m src.cli run-ecosystem      # Run ecosystem simulation
    python -m src.cli serve              # Start API server
    python -m src.cli test               # Run all tests
"""

import argparse
import sys
import os
from typing import Optional


def run_tiger_lifestory(
    prompt: str = "给我一个老虎的生命历程",
    steps: int = 200,
    video: bool = False
):
    """Run a single tiger life story simulation"""
    from .life_story_api import LifestoryAPI

    print(f"🐅 Running life story simulation: {prompt}")
    print(f"⚡ Max steps: {steps}")
    print(f"🎬 Generate video: {video}")

    result = LifestoryAPI.generate(
        prompt=prompt,
        output_video_path="output/tiger_life.mp4" if video else "",
        max_steps=steps,
        render_video=video
    )

    print("\n" + "=" * 50)
    print("📊 Simulation Complete")
    print("=" * 50)
    print(result['summary'])
    print(f"\n⏱️  Runtime: {result['metadata']['runtime_seconds']}s")
    print(f"👣 Steps executed: {result['metadata']['simulation_steps']}")

    return result


def run_ecosystem(
    tigers: int = 1,
    deer: int = 5,
    steps: int = 500,
    show_progress: bool = True
):
    """Run multi-agent ecosystem simulation"""
    from .ecosystem import EcosystemSimulator

    print(f"🌍 Running ecosystem simulation")
    print(f"🐅 Tigers: {tigers}")
    print(f"🦌 Deer: {deer}")
    print(f"⚡ Max steps: {steps}")

    sim = EcosystemSimulator(n_tigers=tigers, n_deer=deer)
    history = sim.run(max_steps=steps)

    print("\n" + "=" * 50)
    print("🌍 Ecosystem Simulation Complete")
    print("=" * 50)
    print(sim.get_summary())

    if history:
        final = history[-1]
        print(f"\n📈 Population dynamics:")
        print(f"   Initial: {tigers + deer} animals")
        print(f"   Final: {final['alive']} animals")
        print(f"   Survival rate: {final['alive'] / (tigers + deer) * 100:.1f}%")

    return sim


def run_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server"""
    from .api import run_server
    run_server(host=host, port=port, reload=reload)


def run_tests(verbose: bool = False):
    """Run the test suite"""
    import pytest

    args = ["tests/"]
    if verbose:
        args.append("-v")

    print("🧪 Running test suite...")
    result = pytest.main(args)

    if result == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Some tests failed (code: {result})")

    return result


def show_demo():
    """Show a quick demo of ecosystem simulation"""
    print("🎮 Active Inference World Simulator Demo")
    print("=" * 50)
    print()

    # Run short ecosystem demo
    sim = run_ecosystem(tigers=1, deer=3, steps=100)

    print("\n" + "=" * 50)
    print("💡 Next steps:")
    print("   python -m src.cli run-tiger --video")
    print("   python -m src.cli run-ecosystem --tigers 2 --deer 10")
    print("   python -m src.cli serve")


def main():
    parser = argparse.ArgumentParser(
        description="Active Inference World Simulator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli run-tiger
  python -m src.cli run-ecosystem --tigers 2 --deer 8
  python -m src.cli serve --port 8000
  python -m src.cli test
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run tiger life story
    tiger_parser = subparsers.add_parser("run-tiger", help="Run tiger life story simulation")
    tiger_parser.add_argument("--prompt", type=str, default="给我一个老虎的生命历程", help="Natural language prompt")
    tiger_parser.add_argument("--steps", type=int, default=200, help="Max simulation steps")
    tiger_parser.add_argument("--video", action="store_true", help="Generate video output")

    # Run ecosystem
    eco_parser = subparsers.add_parser("run-ecosystem", help="Run ecosystem simulation")
    eco_parser.add_argument("--tigers", type=int, default=1, help="Number of tigers")
    eco_parser.add_argument("--deer", type=int, default=5, help="Number of deer")
    eco_parser.add_argument("--steps", type=int, default=500, help="Max simulation steps")

    # API server
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve_parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")

    # Test
    test_parser = subparsers.add_parser("test", help="Run test suite")
    test_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Demo
    subparsers.add_parser("demo", help="Show quick demo")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "run-tiger":
        run_tiger_lifestory(prompt=args.prompt, steps=args.steps, video=args.video)

    elif args.command == "run-ecosystem":
        run_ecosystem(tigers=args.tigers, deer=args.deer, steps=args.steps)

    elif args.command == "serve":
        run_api_server(host=args.host, port=args.port, reload=args.reload)

    elif args.command == "test":
        run_tests(verbose=args.verbose)

    elif args.command == "demo":
        show_demo()


if __name__ == "__main__":
    main()
