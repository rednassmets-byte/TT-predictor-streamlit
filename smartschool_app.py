"""
Smartschool Rapport Calculator - Streamlit App
Berekent gewogen gemiddelde op basis van uren per vak
"""

import streamlit as st
from smartschool import Smartschool, AppCredentials
from smartschool.results import Results
from typing import Dict, List

# Gewichten per vak (gebaseerd op uren)
VAKKEN_UREN = {
    "godsdienst": 2.00,
    "Nederlands": 4.00,
    "geschiedenis": 0.00,
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
    """Bereken gewogen gemiddelde op basis van uren"""
    totaal_gewogen = 0.0
    totaal_uren = 0.0
    details = []
    
    for vak, (jouw_punten, max_punten) in resultaten.items():
        uren = VAKKEN_UREN.get(vak, 0.0)
        
        if uren == 0:
            continue
        
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
        "details": sorted(details, key=lambda x: x["vak"])
    }

def main():
    st.set_page_config(
        page_title="Smartschool Rapport Calculator",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Smartschool Rapport Calculator")
    st.markdown("Bereken je gewogen gemiddelde op basis van uren per vak")
    
    # Sidebar voor credentials
    with st.sidebar:
        st.header("🔐 Login Gegevens")
        
        username = st.text_input("Gebruikersnaam", placeholder="jouw.naam")
        password = st.text_input("Wachtwoord", type="password")
        school_url = st.text_input("School URL", value="smcb.smartschool.be")
        
        st.info("ℹ️ Voor leerlingen: vul je geboortedatum in als MFA")
        mfa_code = st.text_input("Geboortedatum", placeholder="dd/mm/jjjj", help="Vul je geboortedatum in (formaat: dd/mm/jjjj)")
        
        login_button = st.button("📥 Punten Ophalen", type="primary", use_container_width=True)
        
        st.divider()
        st.caption("Je gegevens worden niet opgeslagen")
    
    # Main content
    if login_button:
        if not username or not password:
            st.error("⚠️ Vul je gebruikersnaam en wachtwoord in")
            return
        
        with st.spinner("Verbinden met Smartschool..."):
            try:
                # Maak credentials
                creds = AppCredentials(
                    username=username,
                    password=password,
                    main_url=school_url,
                    mfa=mfa_code if mfa_code else ""
                )
                
                # Maak sessie
                session = Smartschool(creds=creds)
                
                st.success("✅ Verbonden met Smartschool!")
                
                st.info("""
                ℹ️ **Let op:** De Smartschool API werkt niet altijd betrouwbaar.
                Als er vakken zijn zonder resultaten, kan de API een fout geven.
                
                **Gebruik daarom de handmatige invoer hieronder:**
                """)
                
                # Handmatige invoer sectie
                st.subheader("✏️ Vul je punten handmatig in")
                
                resultaten = {}
                
                # Maak twee kolommen voor de vakken
                vakken_list = [(vak, uren) for vak, uren in VAKKEN_UREN.items() if uren > 0]
                
                for idx, (vak, uren) in enumerate(vakken_list):
                    with st.expander(f"{vak} ({uren} uur)"):
                        cols = st.columns([2, 2])
                        
                        with cols[0]:
                            jouw_punten = st.number_input(
                                "Jouw punten",
                                min_value=0.0,
                                max_value=200.0,
                                value=0.0,
                                step=0.5,
                                key=f"manual_jouw_{vak}"
                            )
                        
                        with cols[1]:
                            max_punten = st.number_input(
                                "Max punten",
                                min_value=0.0,
                                max_value=200.0,
                                value=100.0,
                                step=0.5,
                                key=f"manual_max_{vak}"
                            )
                        
                        if jouw_punten > 0:
                            resultaten[vak] = (jouw_punten, max_punten)
                            percentage = bereken_percentage(jouw_punten, max_punten)
                            st.success(f"✓ {percentage:.1f}%")
                
                if not resultaten:
                    st.warning("⚠️ Vul minstens één vak in om te berekenen")
                    return
                
                # Bereken gewogen gemiddelde
                berekening = bereken_gewogen_gemiddelde(resultaten)
                
                # Toon gewogen gemiddelde prominent
                st.divider()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Gewogen Gemiddelde", f"{berekening['gemiddelde']:.2f}%")
                
                with col2:
                    st.metric("📚 Totaal Uren", f"{berekening['totaal_uren']:.2f}")
                
                with col3:
                    st.metric("📝 Aantal Vakken", len(berekening['details']))
                
                st.divider()
                
                # Toon details in tabel
                st.subheader("📋 Details per Vak")
                
                # Maak data voor tabel
                table_data = []
                for detail in berekening["details"]:
                    table_data.append({
                        "Vak": detail["vak"],
                        "Uren": f"{detail['uren']:.2f}",
                        "Jouw Punten": f"{detail['jouw_punten']:.1f}",
                        "Totaal Punten": f"{detail['max_punten']:.1f}",
                        "Percentage": f"{detail['percentage']:.2f}%"
                    })
                
                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Toon vakken die niet meetellen
                niet_meetellende_vakken = [vak for vak, uren in VAKKEN_UREN.items() if uren == 0]
                if niet_meetellende_vakken:
                    st.info(f"ℹ️ Vakken die niet meetellen: {', '.join(niet_meetellende_vakken)}")
                
            except Exception as e:
                st.error(f"❌ Fout bij verbinden: {str(e)}")
                st.info("""
                **Mogelijke oorzaken:**
                - Verkeerde gebruikersnaam of wachtwoord
                - School URL is niet correct
                - Smartschool library niet geïnstalleerd
                
                **Installeer eerst:** `pip install smartschool`
                """)
    
    else:
        # Toon instructies
        st.info("""
        ### 👋 Welkom!
        
        Vul je Smartschool gegevens in de sidebar in en klik op **Punten Ophalen**.
        
        **Hoe het werkt:**
        - Je punten worden opgehaald via de Smartschool API
        - Het gewogen gemiddelde wordt berekend op basis van uren per vak
        - Vakken met 0 uren tellen niet mee
        """)
        
        # Toon vakken configuratie
        with st.expander("⚙️ Vakken Configuratie"):
            st.write("Uren per vak:")
            config_data = [
                {"Vak": vak, "Uren": uren, "Telt mee": "✅" if uren > 0 else "❌"}
                for vak, uren in sorted(VAKKEN_UREN.items())
            ]
            st.dataframe(config_data, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
