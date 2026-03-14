"""
Smartschool Rapport Calculator
Berekent gewogen gemiddelde op basis van uren per vak
"""

from smartschool import Smartschool, PathCredentials
import os
from typing import Dict

# Gewichten per vak (gebaseerd op uren)
# Zet uren op 0 als je een vak niet wilt meetellen
VAKKEN_UREN = {
    "godsdienst": 2.00,
    "Nederlands": 4.00,
    "geschiedenis": 0.00,  # Telt niet mee
    "informatica": 2.00,
    "Aarderijskkunde": 2.00,
    "Wiskunde": 6.00,
    "LO": 2.00,
    "Bouwstenen Frans": 1.20,
    "fysica": 2.00,
    "Chemie": 2.00,
    "Biologie": 2.00,
    "bouwstenen Engels": 1.20,
    "taalvaardigheid engels": 1.80,
    "taalvaardigheid frans": 1.80,
}

def bereken_percentage(jouw_punten: float, totaal_punten: float) -> float:
    """Bereken percentage behaald"""
    if totaal_punten == 0:
        return 0.0
    return (jouw_punten / totaal_punten) * 100

def bereken_gewogen_gemiddelde(resultaten: Dict[str, tuple]) -> dict:
    """
    Bereken gewogen gemiddelde op basis van uren
    
    Args:
        resultaten: Dict met vaknaam -> (jouw_punten, totaal_punten)
    
    Returns:
        Dict met berekeningen en details
    """
    totaal_gewogen = 0.0
    totaal_uren = 0.0
    details = []
    
    for vak, (jouw_punten, max_punten) in resultaten.items():
        # Check of vak in configuratie staat
        uren = VAKKEN_UREN.get(vak, 0.0)
        
        if uren == 0:
            continue  # Skip vakken met 0 uren
        
        percentage = bereken_percentage(jouw_punten, max_punten)
        gewogen_score = percentage * uren
        
        totaal_gewogen += gewogen_score
        totaal_uren += uren
        
        details.append({
            "vak": vak,
            "uren": uren,
            "jouw_punten": jouw_punten,
            "max_punten": max_punten,
            "percentage": percentage
        })
    
    gemiddelde = totaal_gewogen / totaal_uren if totaal_uren > 0 else 0.0
    
    return {
        "gemiddelde": gemiddelde,
        "totaal_uren": totaal_uren,
        "details": details
    }

def main():
    # Credentials ophalen (gebruik environment variables voor veiligheid)
    username = os.getenv("SMARTSCHOOL_USER", "jouw_gebruiker")
    password = os.getenv("SMARTSCHOOL_PASS", "jouw_wachtwoord")
    school_url = os.getenv("SMARTSCHOOL_URL", "jouwschool.smartschool.be")
    
    print("Verbinden met Smartschool...")
    
    try:
        # Maak sessie
        session = Smartschool(PathCredentials(
            username=username,
            password=password,
            main_url=school_url
        ))
        
        print("✓ Verbonden met Smartschool\n")
        
        # Haal resultaten op
        results = session.results
        
        # Converteer naar dict formaat
        resultaten = {}
        for r in results:
            # Pas aan op basis van jouw API structuur
            vak_naam = r.name
            score = float(r.score) if r.score else 0.0
            max_score = float(r.max_score) if hasattr(r, 'max_score') else 100.0
            
            resultaten[vak_naam] = (score, max_score)
        
        # Bereken gewogen gemiddelde
        berekening = bereken_gewogen_gemiddelde(resultaten)
        
        # Toon resultaten
        print("=" * 80)
        print(f"{'Vak':<25} {'Uren':<8} {'Punten':<15} {'Percentage':<12}")
        print("=" * 80)
        
        for detail in berekening["details"]:
            punten_str = f"{detail['jouw_punten']:.1f}/{detail['max_punten']:.1f}"
            print(f"{detail['vak']:<25} {detail['uren']:<8.2f} {punten_str:<15} {detail['percentage']:<12.2f}%")
        
        print("=" * 80)
        print(f"Totaal uren: {berekening['totaal_uren']:.2f}")
        print(f"\n🎯 GEWOGEN GEMIDDELDE: {berekening['gemiddelde']:.2f}%")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Fout: {e}")
        print("\nTips:")
        print("- Controleer je credentials")
        print("- Zorg dat de smartschool library geïnstalleerd is: pip install smartschool")
        print("- Check of de school URL correct is")

if __name__ == "__main__":
    main()
