"""
Smartschool Rapport Calculator - Handmatige Invoer
Bereken gewogen gemiddelde op basis van uren per vak
"""

import streamlit as st
from typing import Dict

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
    
    st.info("""
    ℹ️ **Tip:** Vul alleen de vakken in waar je al punten voor hebt gekregen.
    Vakken die je leeg laat worden niet meegeteld in de berekening.
    """)
    
    # Initialiseer session state voor punten
    if 'punten' not in st.session_state:
        st.session_state.punten = {}
    
    # Toon vakken met invoervelden
    st.subheader("📝 Vul je punten in")
    
    col1, col2 = st.columns(2)
    
    resultaten = {}
    
    for idx, (vak, uren) in enumerate(VAKKEN_UREN.items()):
        if uren == 0:
            continue  # Skip vakken met 0 uren
        
        with col1 if idx % 2 == 0 else col2:
            st.markdown(f"**{vak}** ({uren} uur)")
            
            cols = st.columns([2, 2, 1])
            
            with cols[0]:
                jouw_punten = st.number_input(
                    "Jouw punten",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.5,
                    key=f"jouw_{vak}",
                    label_visibility="collapsed"
                )
            
            with cols[1]:
                max_punten = st.number_input(
                    "Max punten",
                    min_value=0.0,
                    max_value=100.0,
                    value=100.0,
                    step=0.5,
                    key=f"max_{vak}",
                    label_visibility="collapsed"
                )
            
            with cols[2]:
                if jouw_punten > 0 or max_punten != 100:
                    percentage = bereken_percentage(jouw_punten, max_punten)
                    st.metric("", f"{percentage:.1f}%")
            
            if jouw_punten > 0:
                resultaten[vak] = (jouw_punten, max_punten)
    
    st.divider()
    
    # Bereken knop
    if st.button("🎯 Bereken Gewogen Gemiddelde", type="primary", use_container_width=True):
        if not resultaten:
            st.warning("⚠️ Vul eerst je punten in!")
        else:
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
    
    # Sidebar met info
    with st.sidebar:
        st.header("ℹ️ Informatie")
        
        st.markdown("""
        ### Hoe werkt het?
        
        1. Vul je punten in per vak
        2. Klik op "Bereken Gewogen Gemiddelde"
        3. Zie je resultaat!
        
        ### Gewichten
        Elk vak telt mee op basis van het aantal uren:
        """)
        
        config_data = [
            {"Vak": vak, "Uren": uren}
            for vak, uren in sorted(VAKKEN_UREN.items())
            if uren > 0
        ]
        st.dataframe(config_data, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
