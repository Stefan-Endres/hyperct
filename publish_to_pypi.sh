#!/bin/bash
# publish_to_pypi.sh - Script to publish hyperct package to PyPI
#
# Usage:
#   ./publish_to_pypi.sh [--test-only] [--skip-tests] [--skip-tag]
#
# Options:
#   --test-only   Upload to TestPyPI instead of production PyPI
#   --skip-tests  Skip running the test suite
#   --skip-tag    Skip creating a git tag
#
# Prerequisites:
#   1. Install required tools:
#      pip install --upgrade pip build twine
#
#   2. Configure PyPI credentials (one of):
#      - Use API tokens in ~/.pypirc:
#        [pypi]
#        username = __token__
#        password = pypi-AgEIcHlwaS5vcmc...
#
#        [testpypi]
#        username = __token__
#        password = pypi-AgENdGVzdC5weXBpLm9yZw...
#
#      - Or set environment variables:
#        export TWINE_USERNAME=__token__
#        export TWINE_PASSWORD=pypi-AgEIcHlwaS5vcmc...

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
TEST_ONLY=false
SKIP_TESTS=false
SKIP_TAG=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-tag)
            SKIP_TAG=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--test-only] [--skip-tests] [--skip-tag]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PyPI Publishing Script for hyperct   ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Extract version from setup.py
VERSION=$(python -c "import re; content = open('setup.py').read(); print(re.search(r\"version=['\\\"]([^'\\\"]+)['\\\"]|version='([^']+)'\", content).group(1) or re.search(r\"version=['\\\"]([^'\\\"]+)['\\\"]|version='([^']+)'\", content).group(2))")
echo -e "${BLUE}Current version:${NC} ${GREEN}${VERSION}${NC}"
echo ""

# 1. Check git status
echo -e "${YELLOW}[1/7] Checking git status...${NC}"
if [[ -n $(git status --porcelain) ]]; then
    echo -e "${RED}Error: Working directory is not clean.${NC}"
    echo "Please commit or stash your changes before publishing."
    git status --short
    exit 1
fi
echo -e "${GREEN}✓ Working directory is clean${NC}"
echo ""

# 2. Check if version tag already exists
if [ "$SKIP_TAG" = false ]; then
    echo -e "${YELLOW}[2/7] Checking if version tag exists...${NC}"
    if git rev-parse "v${VERSION}" >/dev/null 2>&1; then
        echo -e "${RED}Error: Git tag v${VERSION} already exists.${NC}"
        echo "Please update the version in setup.py before publishing."
        exit 1
    fi
    echo -e "${GREEN}✓ Version tag v${VERSION} does not exist${NC}"
else
    echo -e "${YELLOW}[2/7] Skipping version tag check${NC}"
fi
echo ""

# 3. Run tests
if [ "$SKIP_TESTS" = false ]; then
    echo -e "${YELLOW}[3/7] Running tests...${NC}"
    if ! python -m pytest hyperct/tests/ -q --import-mode=importlib --benchmark-skip; then
        echo -e "${RED}Error: Tests failed. Please fix before publishing.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${YELLOW}[3/7] Skipping tests${NC}"
fi
echo ""

# 4. Clean previous builds
echo -e "${YELLOW}[4/7] Cleaning previous builds...${NC}"
rm -rf build/ dist/ *.egg-info hyperct.egg-info/
echo -e "${GREEN}✓ Cleaned build artifacts${NC}"
echo ""

# 5. Build distribution packages
echo -e "${YELLOW}[5/7] Building distribution packages...${NC}"
python -m build
echo -e "${GREEN}✓ Built distribution packages${NC}"
ls -lh dist/
echo ""

# 6. Check the distribution
echo -e "${YELLOW}[6/7] Checking distribution with twine...${NC}"
python -m twine check dist/*
echo -e "${GREEN}✓ Distribution check passed${NC}"
echo ""

# 7. Upload to PyPI
echo -e "${YELLOW}[7/7] Uploading to PyPI...${NC}"
if [ "$TEST_ONLY" = true ]; then
    echo -e "${BLUE}Uploading to TestPyPI...${NC}"
    python -m twine upload --repository testpypi dist/*
    echo ""
    echo -e "${GREEN}✓ Successfully uploaded to TestPyPI!${NC}"
    echo ""
    echo -e "${BLUE}To test the package, run:${NC}"
    echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hyperct==${VERSION}"
else
    echo -e "${BLUE}Uploading to production PyPI...${NC}"
    read -p "Are you sure you want to upload to production PyPI? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo -e "${RED}Upload cancelled.${NC}"
        exit 1
    fi

    python -m twine upload dist/*
    echo ""
    echo -e "${GREEN}✓ Successfully uploaded to PyPI!${NC}"

    # Create git tag
    if [ "$SKIP_TAG" = false ]; then
        echo ""
        echo -e "${YELLOW}Creating git tag v${VERSION}...${NC}"
        git tag -a "v${VERSION}" -m "Release version ${VERSION}"
        echo -e "${GREEN}✓ Created tag v${VERSION}${NC}"
        echo ""
        echo -e "${BLUE}Don't forget to push the tag:${NC}"
        echo "  git push origin v${VERSION}"
    fi

    echo ""
    echo -e "${BLUE}Package is now available at:${NC}"
    echo "  https://pypi.org/project/hyperct/${VERSION}/"
    echo ""
    echo -e "${BLUE}Users can install it with:${NC}"
    echo "  pip install hyperct==${VERSION}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Publishing completed successfully!   ${NC}"
echo -e "${GREEN}========================================${NC}"