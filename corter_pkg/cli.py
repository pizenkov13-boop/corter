#!/usr/bin/env python3
"""
Corter CLI - Command Line Interface
Run ML optimization from the terminal
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point"""
    # Import here to avoid circular imports
    import corter
    
    parser = argparse.ArgumentParser(
        description='Corter - Autonomous ML Optimization Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  corter run data.csv                    # Run with default config.yaml
  corter run data.csv -c custom.yaml     # Run with custom config
  corter run data.csv --web              # Run with web dashboard
  corter init                            # Create default config.yaml
  corter version                         # Show version info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run ML optimization')
    run_parser.add_argument('data', type=str, help='Path to data CSV file')
    run_parser.add_argument('-c', '--config', type=str, default='config.yaml',
                           help='Path to config YAML file (default: config.yaml)')
    run_parser.add_argument('--web', action='store_true',
                           help='Start web dashboard alongside optimization')
    run_parser.add_argument('--output', type=str, default='results.json',
                           help='Path to save results JSON (default: results.json)')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Create default config.yaml')
    init_parser.add_argument('-o', '--output', type=str, default='config.yaml',
                            help='Output path for config file (default: config.yaml)')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Start web dashboard only')
    web_parser.add_argument('-p', '--port', type=int, default=5000,
                           help='Port to run web server (default: 5000)')
    web_parser.add_argument('--host', type=str, default='127.0.0.1',
                           help='Host to bind to (default: 127.0.0.1)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Handle commands
    if args.command == 'run':
        run_optimization(args)
    elif args.command == 'init':
        create_default_config(args)
    elif args.command == 'version':
        show_version()
    elif args.command == 'web':
        start_web_dashboard(args)


def run_optimization(args):
    """Run ML optimization"""
    import json
    from corter import Corter
    
    print("⚡ Corter - Autonomous ML Optimization")
    print("=" * 60)
    
    # Check if files exist
    if not Path(args.data).exists():
        print(f"❌ Error: Data file not found: {args.data}")
        sys.exit(1)
    
    if not Path(args.config).exists():
        print(f"❌ Error: Config file not found: {args.config}")
        print(f"💡 Tip: Run 'corter init' to create a default config")
        sys.exit(1)
    
    print(f"📊 Data: {args.data}")
    print(f"⚙️  Config: {args.config}")
    print("=" * 60)
    print()
    
    # Start web dashboard if requested
    if args.web:
        import threading
        import web_ui
        print("🌐 Starting web dashboard at http://127.0.0.1:5000")
        web_thread = threading.Thread(target=lambda: web_ui.run_server(debug=False), daemon=True)
        web_thread.start()
        print()
    
    try:
        # Load and run
        core = Corter.from_yaml(args.config)
        print("🚀 Starting optimization...")
        print()
        
        result = core.run(args.data)
        
        # Print results
        print()
        print("=" * 60)
        print("✅ Optimization Complete!")
        print("=" * 60)
        print(f"🏆 Best CV Score: {result['best_cv_score']:.4f}")
        print(f"📈 Best Parameters:")
        for key, value in result['best_params'].items():
            print(f"   • {key}: {value}")
        
        print()
        print("💡 Insights:")
        for insight in result['insights']:
            print(f"   • {insight}")
        
        # Save results
        with open(args.output, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_result = {
                'best_cv_score': float(result['best_cv_score']),
                'best_params': result['best_params'],
                'insights': result['insights']
            }
            json.dump(json_result, f, indent=2)
        
        print()
        print(f"💾 Results saved to: {args.output}")
        
        if args.web:
            print()
            print("🌐 Web dashboard still running at http://127.0.0.1:5000")
            print("   Press Ctrl+C to exit")
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n👋 Shutting down...")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


def create_default_config(args):
    """Create default config.yaml"""
    default_config = """# Corter Configuration
task: classification
target_column: target

model:
  name: random_forest
  params:
    n_estimators: 100

hpo:
  strategy: random
  n_trials: 24
  parallel_trials: 4
  enable_early_stop: true
  patience: 5
  min_delta: 0.001
  cv_folds: 5
  scoring: accuracy

xai:
  use_shap: true
  shap_sample_size: 100
  top_k_features: 10
  permutation_repeats: 8

tui:
  show_live: true
  refresh_hz: 4
"""
    
    output_path = Path(args.output)
    
    if output_path.exists():
        response = input(f"⚠️  {args.output} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            sys.exit(0)
    
    with open(output_path, 'w') as f:
        f.write(default_config)
    
    print(f"✅ Created default config: {args.output}")
    print()
    print("📝 Next steps:")
    print(f"   1. Edit {args.output} to customize settings")
    print(f"   2. Run: corter run data.csv")


def show_version():
    """Show version information"""
    from corter_pkg import __version__
    print(f"Corter v{__version__}")
    print("Autonomous ML Optimization Framework")
    print()
    print("GitHub: https://github.com/pizenkov13-boop/Corter")


def start_web_dashboard(args):
    """Start web dashboard only"""
    import web_ui
    
    print("🚀 Starting Corter Web Dashboard")
    print(f"📊 Dashboard: http://{args.host}:{args.port}")
    print("🎬 Demo mode: ENABLED")
    print()
    print("Press Ctrl+C to stop")
    
    try:
        web_ui.run_server(host=args.host, port=args.port, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")


if __name__ == '__main__':
    main()
