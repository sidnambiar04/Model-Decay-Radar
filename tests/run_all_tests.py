"""
Master Test Suite Runner — Model Decay Radar
============================================
Discovers and executes unit and integration test suites for all stages:
  - test_stage1.py
  - test_stage2_predict_api.py
  - test_stage3_vae_kl.py
  - test_stage4_wasserstein_ks.py
  - test_stage5_adwin_ddm.py
  - test_stage6_uncertainty.py
  - test_stage7_fusion.py
  - test_stage8_mhs.py
  - test_stage9_root_cause.py
  - test_stage10_multiclassifier.py
  - test_stage11_retraining_smote.py
  - test_stage12_validation_gate.py
  - test_stage13_versioning_sqlite.py
  - test_stage14_simulator.py
  - test_stage15_modes.py
  - test_stage16_frontend_api.py
  - test_stage17_model_lifecycle_enhanced.py
  - test_stage18_lifecycle_advanced.py
"""

import sys, os
import unittest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)


def run_master_test_suite():
    print("======================================================================")
    print("   MODEL DECAY RADAR -- MASTER SYSTEM VERIFICATION TEST SUITE   ")
    print("======================================================================")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_modules = [
        "test_stage1",
        "test_stage2_predict_api",
        "test_stage3_vae_kl",
        "test_stage4_wasserstein_ks",
        "test_stage5_adwin_ddm",
        "test_stage6_uncertainty",
        "test_stage7_fusion",
        "test_stage8_mhs",
        "test_stage9_root_cause",
        "test_stage10_multiclassifier",
        "test_stage11_retraining_smote",
        "test_stage12_validation_gate",
        "test_stage13_versioning_sqlite",
        "test_stage14_simulator",
        "test_stage15_modes",
        "test_stage16_frontend_api",
        "test_stage17_model_lifecycle_enhanced",
        "test_stage18_lifecycle_advanced",
    ]

    for mod in test_modules:
        try:
            loaded = loader.loadTestsFromName(mod)
            suite.addTest(loaded)
            print(f"  [+] Loaded test module: {mod}")
        except Exception as e:
            print(f"  [-] Failed to load {mod}: {e}")

    print("----------------------------------------------------------------------")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\nALL MASTER VERIFICATION TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print(f"\nMASTER VERIFICATION TESTS FAILED ({len(result.failures)} failures, {len(result.errors)} errors)")
        return 1


if __name__ == "__main__":
    sys.exit(run_master_test_suite())
