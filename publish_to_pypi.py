#!/usr/bin/env python
"""
publish_to_pypi.py - Cross-platform script to publish hyperct package to PyPI

Usage:
    python publish_to_pypi.py [--test-only] [--skip-tests] [--skip-tag]

Options:
    --test-only   Upload to TestPyPI instead of production PyPI
    --skip-tests  Skip running the test suite
    --skip-tag    Skip creating a git tag

Prerequisites:
    pip install --upgrade pip build twine
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

    @classmethod
    def disable(cls):
        """Disable colors (for Windows without ANSI support)"""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.NC = ''


# Disable colors on Windows if not supported
if sys.platform == 'win32' and not os.environ.get('ANSICON'):
    Colors.disable()


def print_colored(message, color=Colors.NC, **kwargs):
    """Print colored message"""
    print(f"{color}{message}{Colors.NC}", **kwargs)


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command"""
    if isinstance(cmd, str):
        cmd = cmd.split()

    result = subprocess.run(
        cmd,
        check=check,
        capture_output=capture_output,
        text=True
    )
    return result


def get_version():
    """Extract version from setup.py"""
    with open('setup.py', 'r') as f:
        content = f.read()

    match = re.search(r"version=['\"]([^'\"]+)['\"]", content)
    if not match:
        print_colored("Error: Could not find version in setup.py", Colors.RED)
        sys.exit(1)

    return match.group(1)


def check_git_status():
    """Check if working directory is clean"""
    result = run_command('git status --porcelain', capture_output=True)
    return result.stdout.strip() == ''


def tag_exists(tag):
    """Check if a git tag exists"""
    result = run_command(
        f'git rev-parse {tag}',
        check=False,
        capture_output=True
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Publish hyperct package to PyPI'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Upload to TestPyPI instead of production PyPI'
    )
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip running the test suite'
    )
    parser.add_argument(
        '--skip-tag',
        action='store_true',
        help='Skip creating a git tag'
    )
    args = parser.parse_args()

    print_colored("=" * 40, Colors.BLUE)
    print_colored("  PyPI Publishing Script for hyperct", Colors.BLUE)
    print_colored("=" * 40, Colors.BLUE)
    print()

    # Extract version
    version = get_version()
    print_colored(f"Current version: ", Colors.BLUE, end='')
    print_colored(version, Colors.GREEN)
    print()

    # 1. Check git status
    print_colored("[1/7] Checking git status...", Colors.YELLOW)
    if not check_git_status():
        print_colored("Error: Working directory is not clean.", Colors.RED)
        print("Please commit or stash your changes before publishing.")
        run_command('git status --short', check=False)
        sys.exit(1)
    print_colored("✓ Working directory is clean", Colors.GREEN)
    print()

    # 2. Check if version tag already exists
    if not args.skip_tag:
        print_colored("[2/7] Checking if version tag exists...", Colors.YELLOW)
        if tag_exists(f"v{version}"):
            print_colored(f"Error: Git tag v{version} already exists.", Colors.RED)
            print("Please update the version in setup.py before publishing.")
            sys.exit(1)
        print_colored(f"✓ Version tag v{version} does not exist", Colors.GREEN)
    else:
        print_colored("[2/7] Skipping version tag check", Colors.YELLOW)
    print()

    # 3. Run tests
    if not args.skip_tests:
        print_colored("[3/7] Running tests...", Colors.YELLOW)
        try:
            run_command([sys.executable, '-m', 'pytest', 'hyperct/tests/',
                        '-q', '--import-mode=importlib', '--benchmark-skip'])
            print_colored("✓ All tests passed", Colors.GREEN)
        except subprocess.CalledProcessError:
            print_colored("Error: Tests failed. Please fix before publishing.", Colors.RED)
            sys.exit(1)
    else:
        print_colored("[3/7] Skipping tests", Colors.YELLOW)
    print()

    # 4. Clean previous builds
    print_colored("[4/7] Cleaning previous builds...", Colors.YELLOW)
    for dir_name in ['build', 'dist', 'hyperct.egg-info']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    # Clean any .egg-info directories
    for item in os.listdir('.'):
        if item.endswith('.egg-info'):
            shutil.rmtree(item)
    print_colored("✓ Cleaned build artifacts", Colors.GREEN)
    print()

    # 5. Build distribution packages
    print_colored("[5/7] Building distribution packages...", Colors.YELLOW)
    run_command([sys.executable, '-m', 'build'])
    print_colored("✓ Built distribution packages", Colors.GREEN)

    # List built packages
    if os.path.exists('dist'):
        for filename in os.listdir('dist'):
            filepath = os.path.join('dist', filename)
            size = os.path.getsize(filepath)
            print(f"  {filename} ({size:,} bytes)")
    print()

    # 6. Check the distribution
    print_colored("[6/7] Checking distribution with twine...", Colors.YELLOW)
    dist_files = glob.glob('dist/*')
    if not dist_files:
        print_colored("Error: No files found in dist/", Colors.RED)
        sys.exit(1)
    run_command([sys.executable, '-m', 'twine', 'check'] + dist_files)
    print_colored("✓ Distribution check passed", Colors.GREEN)
    print()

    # 7. Upload to PyPI
    print_colored("[7/7] Uploading to PyPI...", Colors.YELLOW)

    if args.test_only:
        print_colored("Uploading to TestPyPI...", Colors.BLUE)
        run_command([
            sys.executable, '-m', 'twine', 'upload',
            '--repository', 'testpypi'] + glob.glob('dist/*')
        )
        print()
        print_colored("✓ Successfully uploaded to TestPyPI!", Colors.GREEN)
        print()
        print_colored("To test the package, run:", Colors.BLUE)
        print(f"  pip install --index-url https://test.pypi.org/simple/ "
              f"--extra-index-url https://pypi.org/simple/ hyperct=={version}")
    else:
        print_colored("Uploading to production PyPI...", Colors.BLUE)
        confirm = input("Are you sure you want to upload to production PyPI? (yes/no): ")
        if confirm.lower() != 'yes':
            print_colored("Upload cancelled.", Colors.RED)
            sys.exit(1)

        run_command([sys.executable, '-m', 'twine', 'upload'] + glob.glob('dist/*'))
        print()
        print_colored("✓ Successfully uploaded to PyPI!", Colors.GREEN)

        # Create git tag
        if not args.skip_tag:
            print()
            print_colored(f"Creating git tag v{version}...", Colors.YELLOW)
            run_command(['git', 'tag', '-a', f"v{version}", '-m', f"Release version {version}"])
            print_colored(f"✓ Created tag v{version}", Colors.GREEN)
            print()
            print_colored("Don't forget to push the tag:", Colors.BLUE)
            print(f"  git push origin v{version}")

        print()
        print_colored("Package is now available at:", Colors.BLUE)
        print(f"  https://pypi.org/project/hyperct/{version}/")
        print()
        print_colored("Users can install it with:", Colors.BLUE)
        print(f"  pip install hyperct=={version}")

    print()
    print_colored("=" * 40, Colors.GREEN)
    print_colored("  Publishing completed successfully!", Colors.GREEN)
    print_colored("=" * 40, Colors.GREEN)


if __name__ == '__main__':
    main()