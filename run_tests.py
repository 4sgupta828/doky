#!/usr/bin/env python3
"""
Test runner for all tests in the tests/ directory.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_all_tests():
    """Run all test files in the tests directory."""
    print("🚀 Running All Tests")
    print("=" * 60)
    
    tests_dir = Path(__file__).parent / "tests"
    test_files = list(tests_dir.glob("test_*.py"))
    
    if not test_files:
        print("❌ No test files found in tests/ directory")
        return False
    
    passed = 0
    failed = 0
    
    for test_file in sorted(test_files):
        print(f"\n📋 Running {test_file.name}...")
        print("-" * 40)
        
        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=False,
                text=True,
                check=True
            )
            print(f"✅ {test_file.name} PASSED")
            passed += 1
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {test_file.name} FAILED (exit code {e.returncode})")
            failed += 1
        except Exception as e:
            print(f"💥 {test_file.name} ERROR: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("🎉 All tests PASSED!")
        return True
    else:
        print(f"⚠️ {failed} test(s) failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)