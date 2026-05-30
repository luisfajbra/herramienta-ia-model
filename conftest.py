# conftest.py — project-root pytest configuration
"""Set environment variables required before any test module imports."""
import os

# Prevent segfault when xgboost (LLVM OpenMP) and torch (Intel OpenMP/libiomp5)
# are both loaded in the same process on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Limit OMP threads to avoid conflicts between LLVM-OMP and Intel-OMP runtimes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
