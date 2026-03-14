# ELO Model Gebruik - Overzicht

## Huidige Configuratie

### ELO Model wordt gebruikt voor:

1. **Volwassen spelers met special cases**
   - Categorieën: SEN, VET, etc.
   - Voorwaarden: Grote sprong verwacht (hoge win rate + dominantie) OF NG
   - Bericht: "🎯 Using ELO-enhanced model for special case prediction"

2. **Oudere jeugd met special cases**
   - Categorieën: JUN, J19, J21
   - Voorwaarden: Grote sprong verwacht (hoge win rate + dominantie) OF NG
   - Bericht: "🎯 Using ELO-enhanced model for special case prediction"

### ELO Model wordt NIET gebruikt voor:

1. **Jongere jeugdspelers**
   - Categorieën: BEN, PRE, MIN, CAD
   - Alle klassementen: NG, E6, E4, E2, E0, D6, D4, etc.
   - Reden: V3 filtered model is beter afgestemd op jongere jeugd
   - Deze spelers gebruiken ALTIJD V3 filtered model

## Logica in Code

```python
# Check if younger youth (exclude from ELO)
younger_youth = ['BEN', 'PRE', 'MIN', 'CAD']
is_younger_youth = category in younger_youth

# Use ELO model for older youth (JUN, J19, J21) and adults
if not is_younger_youth and is_special_case(player_data, rank_to_int):
    if elo > 0:
        use_special_model = True  # Use V4 ELO model
```

## Voorbeelden

### ✅ Gebruikt ELO Model:
- SEN speler met NG klassement + ELO 1200
- SEN speler met C4 klassement + hoge win rate + ELO 1100
- VET speler met D2 klassement + hoge win rate + ELO 850
- JUN speler met NG klassement + ELO 600 (special case)
- J19 speler met D0 klassement + hoge win rate + ELO 900 (special case)

### ❌ Gebruikt GEEN ELO Model:
- CAD speler met NG klassement (gebruikt V3 filtered)
- CAD speler met D4 klassement (gebruikt V3 filtered)
- MIN speler met E2 klassement (gebruikt V3 filtered)
- PRE speler met E6 klassement (gebruikt V3 filtered)
- BEN speler met NG klassement (gebruikt V3 filtered)

## Voordelen van deze aanpak

1. **Jongere jeugd krijgt geoptimaliseerde voorspelling**
   - V3 filtered model kent jeugd-specifieke patronen
   - Lagere thresholds voor verbetering
   - Betere handling van snelle ontwikkeling
   - Geldt voor BEN, PRE, MIN, CAD (ook NG)

2. **Oudere jeugd krijgt ELO voordeel**
   - JUN, J19, J21 kunnen ELO model gebruiken voor special cases
   - Deze spelers zijn ouder en hebben meer stabiele ELO
   - Betere voorspelling van grote sprongen

3. **Volwassenen behouden ELO voordeel**
   - Voor special cases (grote sprongen en NG)
   - Betere voorspelling van uitzonderlijke prestaties
   - ELO is betrouwbaarder voor volwassenen (meer matches)

4. **Logische leeftijdsgrens**
   - Jongere jeugd (< 15 jaar) = ALTIJD V3 filtered
   - Oudere jeugd (15-21 jaar) = V3 filtered + V4 ELO voor special cases
   - Volwassenen = V3 regular + V4 ELO voor special cases

## Testen

Test met verschillende spelers:
- CAD speler met NG → Moet V3 filtered gebruiken (GEEN ELO)
- MIN speler met E6 → Moet V3 filtered gebruiken (GEEN ELO)
- JUN speler met NG + special case → Moet ELO model gebruiken
- J19 speler met hoge win rate + special case → Moet ELO model gebruiken
- SEN speler met NG → Moet ELO model gebruiken
- SEN speler met C4 + hoge win rate → Moet ELO model gebruiken
