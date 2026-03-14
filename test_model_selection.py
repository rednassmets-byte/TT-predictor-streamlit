"""
Test script to verify model selection logic
"""

# Test categories
test_cases = [
    ("JUN", "Should use JUN/J19 model"),
    ("J19", "Should use JUN/J19 model"),
    ("BEN", "Should use Filtered model"),
    ("PRE", "Should use Filtered model"),
    ("MIN", "Should use Filtered model"),
    ("CAD", "Should use Filtered model"),
    ("SEN", "Should use Regular model"),
    ("VET", "Should use Regular model"),
    ("HER", "Should use Regular model"),
]

print("=" * 70)
print("MODEL SELECTION LOGIC TEST")
print("=" * 70)

for category, expected in test_cases:
    # Simulate the app's logic
    if category in ["JUN", "J19"]:
        model_type = "JUN/J19 (specialized)"
    elif category in ["BEN", "PRE", "MIN", "CAD"]:
        model_type = "Filtered (youth categories)"
    else:
        model_type = "Regular (adult categories)"
    
    status = "✓" if (
        ("JUN/J19" in model_type and "JUN/J19" in expected) or
        ("Filtered" in model_type and "Filtered" in expected) or
        ("Regular" in model_type and "Regular" in expected)
    ) else "✗"
    
    print(f"{status} {category:4s} -> {model_type:30s} ({expected})")

print("=" * 70)
print("\nModel Selection Summary:")
print("  JUN, J19           -> JUN/J19 specialized model")
print("  BEN, PRE, MIN, CAD -> Filtered youth model")
print("  All others         -> Regular adult model")
print("=" * 70)
