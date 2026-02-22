
def detect_flux_architecture(num_double, num_single, model_type):
    # Logic from generator.py
    is_actually_flux2 = (num_double > 0 and num_single > 0) and not (num_double == 19 and num_single == 38)
    
    if is_actually_flux2:
        if model_type == 'flux':
            return "Flux 2 (Warning: Intent was Flux 1)"
        else:
            return "Flux 2"
    else:
        if model_type == 'flux2':
            return "Flux 1 (Warning: Intent was Flux 2)"
        else:
            return "Flux 1"

def run_tests():
    test_cases = [
        {"d": 19, "s": 38, "type": "flux", "expected": "Flux 1", "name": "Flux 1 File + Flux 1 Intent"},
        {"d": 5, "s": 20, "type": "flux2", "expected": "Flux 2", "name": "Flux 2 File + Flux 2 Intent"},
        {"d": 5, "s": 20, "type": "flux", "expected": "Flux 2 (Warning: Intent was Flux 1)", "name": "Flux 2 File + Flux 1 Intent"},
        {"d": 19, "s": 38, "type": "flux2", "expected": "Flux 1 (Warning: Intent was Flux 2)", "name": "Flux 1 File + Flux 2 Intent"},
    ]

    print("Running Flux Architecture Detection Tests...")
    print("-" * 60)

    pass_count = 0
    for case in test_cases:
        result = detect_flux_architecture(case["d"], case["s"], case["type"])
        status = "PASS" if result == case["expected"] else "FAIL"
        if status == "PASS": pass_count += 1
        print(f"[{status}] {case['name']}: {result}")

    print("-" * 60)
    print(f"Verification {('SUCCESSFUL' if pass_count == len(test_cases) else 'FAILED')} ({pass_count}/{len(test_cases)})")

if __name__ == "__main__":
    run_tests()
