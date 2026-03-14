from smartschool import Smartschool
import inspect

# Check wat er beschikbaar is
print("=== Smartschool attributes ===")
ss_attrs = [attr for attr in dir(Smartschool) if not attr.startswith('_')]
for attr in ss_attrs:
    print(f"  {attr}")

print("\n=== Check for results/reports ===")
if hasattr(Smartschool, 'get_results'):
    print("get_results:", inspect.signature(Smartschool.get_results))
if hasattr(Smartschool, 'get_reports'):
    print("get_reports:", inspect.signature(Smartschool.get_reports))
if hasattr(Smartschool, 'reports'):
    print("reports exists")
if hasattr(Smartschool, 'results'):
    print("results exists")
