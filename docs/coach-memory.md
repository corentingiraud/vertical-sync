# Mémoire du coach (`coach/`)

Vertical Sync sépare trois couches :

| Couche | Rôle | Où | LLM ? |
|--------|------|-----|-------|
| **Sense** | Analyse déterministe des traces FIT, sortie JSON | CLI `vs` + `config/athlete.toml` | non |
| **Reason** | Interprétation, plans, suivis, décisions | LLM (skills Claude Code) | oui |
| **Remember** | Mémoire d'entraînement versionnée en markdown | `coach/` | — |

Le CLI **calcule** (métriques, comparaison au plan, drapeaux d'`analysis.py`).
Le LLM **interprète et décide**, en lisant/écrivant la mémoire `coach/`. Les
heuristiques du CLI sont des *entrées* pour le LLM, pas des verdicts finaux.

`coach/` est **gitignored** : c'est de la donnée personnelle (et de santé). Le
dépôt public livre `templates/` et ce document à la place. Une nouvelle copie du
repo part d'un `coach/` vide et le reconstruit à partir des templates.

## Structure

```
coach/
├── athlete.md          # QUI  — profil persistant (forces, faiblesses, blessures, matériel, perfs)
├── plan/
│   ├── README.md       # vue d'ensemble du bloc + séances types récurrentes
│   └── YYYY-MM-DD.md   # 1 par semaine : frontmatter TOML (cibles) + prose des séances
├── journal/
│   └── YYYY-<race>.md  # QUAND — log daté, plus récent en haut, 1 fichier par bloc
└── races/
    └── <race>.md       # jour-J : cheatsheet + débrief post-course
```

### `athlete.md` — mémoire persistante

Ce qui reste vrai d'un bloc à l'autre : forces, faiblesses, historique blessures,
matériel validé, nutrition testée, perfs de référence. À distinguer de :

- `config/athlete.toml` — les **chiffres déterministes** que le CLI lit (FC max,
  zones, allure seuil, cible de course). Pas de markdown ici.
- `coach/journal/` — le **log daté** des observations.

On met `athlete.md` à jour quand un fait *durable* change (nouvelle perf de
référence, blessure résolue, matériel validé), pas pour une observation du jour.
Template : [`templates/athlete.md`](../templates/athlete.md).

### `plan/` — prospectif

Un fichier markdown par semaine, nommé par la date du lundi (`YYYY-MM-DD.md`).
Le **frontmatter TOML** (entre les barrières `+++`) porte les cibles numériques
que le CLI compare à l'entraînement réel ; le corps markdown est de la prose
libre (séances, intensités, consignes) pour l'humain et le LLM.

Le CLI ne lit **que** le frontmatter (`config.py:_load_plan_weeks()`). Champs :
`week`, `start`, `end` (YYYYMMDD), `phase`, `target_hours`, `target_dplus`,
`target_sessions`. `plan/README.md` porte la vue d'ensemble du bloc et les
séances types récurrentes. Template : [`templates/plan-week.md`](../templates/plan-week.md).

### `journal/` — chronologique

Un fichier par bloc d'entraînement (`YYYY-<race>.md`), entrées datées, **la plus
récente en haut**. Le log brut que le LLM relit pour le contexte récent : comment
ça se passe, ce qui a fait mal, ce qui a été décidé et pourquoi. Les faits qui
survivent au bloc sont distillés dans `athlete.md`. Template :
[`templates/journal-entry.md`](../templates/journal-entry.md).

### `races/` — jour-J

Un fichier par course (`<race>.md`) : la cheatsheet jour-J (allure, écran montre,
nutrition, protocoles) et, après la course, le débrief (résultat, conditions,
ce qui a marché). C'est un artefact *généré* par le LLM à partir du plan, du
journal et du profil.

## Boucle de travail typique

1. `vs download` puis `vs week --json` → le CLI produit les métriques de la semaine.
2. Le LLM lit le JSON + `coach/athlete.md` + le journal + la semaine de plan.
3. Il interprète, met à jour le journal (entrée datée), ajuste le plan si besoin.
4. Les faits durables remontent dans `athlete.md`.

Voir `.claude/skills/` pour les commandes coach qui orchestrent cette boucle.
