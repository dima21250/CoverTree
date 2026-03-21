#!/bin/bash
# Build script for CoverTree using CMake
# Supports both C++ executables and Python extension

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}CoverTree Build Script${NC}"
echo -e "${BLUE}========================================${NC}"

# Default values
BUILD_TYPE="Release"
CLEAN=false
PYTHON_BUILD=false
CMAKE_BUILD=true
PARALLEL_JOBS=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --python)
            PYTHON_BUILD=true
            shift
            ;;
        --cmake-only)
            PYTHON_BUILD=false
            shift
            ;;
        --jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --debug       Build in debug mode (default: Release)"
            echo "  --clean       Clean build directories before building"
            echo "  --python      Also build Python extension"
            echo "  --cmake-only  Only build C++ executables (skip Python)"
            echo "  --jobs N      Use N parallel jobs (default: auto-detect)"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${GREEN}Cleaning build directories...${NC}"
    rm -rf build dist *.egg-info
    rm -f covertreec.*.so
    echo -e "${GREEN}✓ Clean complete${NC}"
fi

# CMake build
if [ "$CMAKE_BUILD" = true ]; then
    echo ""
    echo -e "${GREEN}Building C++ executables with CMake...${NC}"
    echo -e "${BLUE}Build type: ${BUILD_TYPE}${NC}"
    echo -e "${BLUE}Parallel jobs: ${PARALLEL_JOBS}${NC}"

    # Create build directory
    mkdir -p build
    cd build

    # Configure
    echo -e "${GREEN}Configuring...${NC}"
    cmake .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE}

    # Build
    echo -e "${GREEN}Building...${NC}"
    cmake --build . -j${PARALLEL_JOBS}

    # Install to dist/
    echo -e "${GREEN}Installing to dist/...${NC}"
    cmake --install . --prefix=../dist

    cd ..

    echo -e "${GREEN}✓ C++ build complete${NC}"
    echo -e "${BLUE}Executables in: dist/bin/${NC}"
fi

# Python build
if [ "$PYTHON_BUILD" = true ]; then
    echo ""
    echo -e "${GREEN}Building Python extension...${NC}"

    # Check if in virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        echo -e "${BLUE}Note: Not in a virtual environment${NC}"
        echo -e "${BLUE}Activate with: source .venv/bin/activate${NC}"
    fi

    # Build with UV
    if command -v uv &> /dev/null; then
        echo -e "${GREEN}Using UV for fast installation...${NC}"
        uv pip install -e . --no-build-isolation --force-reinstall
    else
        echo -e "${BLUE}UV not found, using pip...${NC}"
        pip install -e . --no-build-isolation --force-reinstall
    fi

    echo -e "${GREEN}✓ Python build complete${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build Summary${NC}"
echo -e "${GREEN}========================================${NC}"
if [ "$CMAKE_BUILD" = true ]; then
    echo -e "${GREEN}C++ executables:${NC}"
    echo -e "  • dist/bin/cover_tree"
    echo -e "  • dist/bin/sg_tree"
fi
if [ "$PYTHON_BUILD" = true ]; then
    echo -e "${GREEN}Python module:${NC}"
    echo -e "  • covertree (installed)"
fi
echo ""
echo -e "${GREEN}Test commands:${NC}"
if [ "$CMAKE_BUILD" = true ]; then
    echo -e "  ${BLUE}./dist/bin/cover_tree <train.dat> <test.dat>${NC}"
fi
if [ "$PYTHON_BUILD" = true ]; then
    echo -e "  ${BLUE}python test_sanity.py${NC}"
    echo -e "  ${BLUE}python test_clustering.py${NC}"
fi
echo -e "${GREEN}========================================${NC}"
