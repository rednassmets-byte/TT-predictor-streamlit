"""
Test script om te zien welke data we kunnen ophalen
LET OP: Vul je eigen credentials in!
"""
from smartschool import Smartschool, AppCredentials
from smartschool.results import Results
from smartschool.reports import Reports

# Vul hier je gegevens in
USERNAME = input("Gebruikersnaam: ")
PASSWORD = input("Wachtwoord: ")
MFA = input("Geboortedatum (dd/mm/jjjj): ")

try:
    creds = AppCredentials(
        username=USERNAME,
        password=PASSWORD,
        main_url="smcb.smartschool.be",
        mfa=MFA
    )
    
    session = Smartschool(creds=creds)
    print("✅ Verbonden met Smartschool!")
    
    # Test Results
    print("\n=== Test Results ===")
    try:
        results_obj = Results(session)
        results = list(results_obj)
        print(f"Aantal resultaten: {len(results)}")
        if results:
            print(f"Eerste resultaat: {results[0]}")
            print(f"Attributes: {dir(results[0])}")
    except Exception as e:
        print(f"❌ Results error: {e}")
    
    # Test Reports
    print("\n=== Test Reports ===")
    try:
        reports_obj = Reports(session)
        reports = list(reports_obj)
        print(f"Aantal rapporten: {len(reports)}")
        if reports:
            print(f"Eerste rapport: {reports[0]}")
            print(f"Attributes: {dir(reports[0])}")
    except Exception as e:
        print(f"❌ Reports error: {e}")
        
except Exception as e:
    print(f"❌ Fout: {e}")
