from smartschool import Smartschool, Credentials, AppCredentials, EnvCredentials
import inspect

print("=== Credentials ===")
print(inspect.signature(Credentials.__init__))
print(Credentials.__doc__)

print("\n=== AppCredentials ===")
print(inspect.signature(AppCredentials.__init__))

print("\n=== EnvCredentials ===")
print(inspect.signature(EnvCredentials.__init__))

print("\n=== Smartschool ===")
print(inspect.signature(Smartschool.__init__))
