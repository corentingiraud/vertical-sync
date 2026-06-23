# Metrics Backlog

Metriques identifiees pour enrichir l'analyse trail running, classees par priorite.

## Implementees

- **Grade-Adjusted Pace (GAP)** — Allure equivalente plat via polynome de Minetti
- **Profil de performance par gradient** — Allure + FC par tranche de pente (-20% a +20%)

## Top priorite

- **Decoupling aerobie ameliore (Pa:HR)** — Ratio pace_normalisee/FC par moitie. Cible < 5% pour readiness ultra. Formule TrainingPeaks : `(first_half_ratio - second_half_ratio) / first_half_ratio * 100` ou ratio = normalized_pace / avg_hr
- **Training Stress Score (rTSS)** — Score unique combinant duree + intensite. `rTSS = (duree_s * IF^2 * 100) / 3600` ou IF = NGP / allure_seuil. Permet PMC (CTL/ATL/TSB) pour piloter charge et taper.
- **Efficiency Factor (EF)** — GAP normalise / FC moyenne. A suivre dans le temps sur sorties similaires. EF qui monte = progression aerobie.

## Haute valeur

- **Performance en descente** — Degradation descente 24.5% > montee sur ultra (etude PMC). Comparer allure descente debut vs fin de sortie longue.
- **Puissance estimee (sans capteur)** — GOVSS : `Power = vitesse * ECOR * masse`. ECOR baseline ~0.98 kJ/kg/km plat. Normalise l'effort sur tout terrain.
- **Pace Variability Index** — Ratio allure normalisee / allure moyenne. Plus c'est haut, plus le pacing est irregulier.

## Bonus (si donnees disponibles)

- **Stride dynamics** — Vertical oscillation (cible 5-10cm), GCT, L/R balance. Tracer degradation sur sorties longues = fatigue musculaire.
- **Categorisation des montees** — Detecter et categoriser les montees (Cat 5 a HC), tracker VAM par type dans le temps.
- **DFA alpha1** — Si RR intervals disponibles. Seuil aerobie a 0.75, anaerobique a 0.50.
- **Critical Speed / Power-Duration curve** — Vitesse critique = max soutenable ~30-60min. D' = reserve au-dessus. Guide prescription intensite.
- **Temperature / altitude correlation** — FIT enregistre la temperature. Correler avec derive FC et degradation allure.

## Sources

- TrainingPeaks: Normalized Graded Pace, Aerobic Decoupling, Efficiency Factor, TSS
- Fellrnr: Grade Adjusted Pace (polynome Minetti)
- PickleTech: Profil gradient individualise (analyse Kilian Jornet)
- Runalyze: GAP, categorisation montees, puissance estimee
- PMC: Downhill sections in trail ultramarathons (etude 16,518 athletes)
- Stryd: ECOR, analyse puissance course
- COROS: Advanced Running Metrics (stride dynamics)
