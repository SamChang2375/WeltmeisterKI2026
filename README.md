# WeltmeisterKI4

WeltmeisterKI4 ist ein datengetriebenes Prognose-System fuer die FIFA WM 2026. Es kombiniert klassische Football-Analytics-Features mit Machine Learning, einem kleinen Deep-Learning-Modell und Monte-Carlo-Simulationen. Ziel ist nicht nur ein einzelner Tipp wie `2:1`, sondern eine vollstaendige Wahrscheinlichkeitsanalyse: Sieg, Remis, Niederlage, erwartete Tore, wahrscheinlichste Scorelines, Gruppenstaende, KO-Pfade, Titelchancen und Ausscheidungswahrscheinlichkeiten.

Das Projekt ist bewusst als Notebook-Workflow gebaut: Daten laden, Modell laden oder trainieren, aktuelle WM-Ergebnisse eintragen, Prognosen neu berechnen.

## Notebook-Varianten

### `WeltmeisterKI4_INFERENCE.ipynb`

Das ist die Nutzer-Version. Sie ist dafuer gedacht, das finale Modell einfach zu laden und Vorhersagen zu erzeugen.

Diese Version:

- trainiert kein neues Modell
- laedt die fertigen Modellartefakte
- funktioniert auch ohne GPU
- aktualisiert Prognosen mit bereits gespielten WM-Spielen
- gibt Zusatzstatistiken fuer Tippspiel-Entscheidungen aus
- simuliert Gruppenphase, KO-Runden und komplette Turnierbaeume

Wenn eine NVIDIA-GPU vorhanden ist, nutzt PyTorch sie automatisch. Wenn keine GPU vorhanden ist, laeuft die Inference auf CPU. Das ist langsamer als CUDA, aber fuer normale Prognosezellen und Monte Carlo grundsaetzlich nutzbar.

### `WeltmeisterKI4_SOTA_Final.ipynb`

Das ist die Trainings- und Research-Version. Dort werden Features gebaut, XGBoost-Modelle trainiert, das Deep-Learning-Modell trainiert, Backtests gerechnet und das finale Hybridmodell gespeichert.

Diese Version ist rechenintensiver und eher fuer Entwicklung, Tuning und Modellvergleich gedacht.

## Modellartefakte

Die Inference-Version erwartet die trainierten Modelle im selben Projektordner. Standardmaessig sucht sie zuerst in:

```text
models/
```

und danach direkt neben dem Notebook.

Ben?tigte Modellfiles:

```text
xgb_goal_models_v2_ensemble.joblib
weltmeisterki_deep_v1.pt
weltmeisterki4_sota_meta.joblib
```

Die Idee ist: Einmal mit dem Trainingsnotebook erzeugen, danach mit dem Inference-Notebook beliebig oft nutzen.

## Datenbasis

WeltmeisterKI4 arbeitet mit kostenlosen und offenen Datenquellen.

### Historische Laenderspiele ab 1980

Die Basis kommt aus historischen internationalen Spielresultaten. Verwendet werden unter anderem:

- Datum
- Heimteam
- Auswaertsteam
- Tore
- Turnier
- Stadt und Land
- neutraler Platz
- Torschuetzen
- Elfmeterinformationen
- Eigentore
- Shootouts

Das Modell lernt nicht nur aus Weltmeisterschaften. Das waeren viel zu wenige Spiele. Stattdessen werden alle Laenderspiele ab 1980 genutzt, wobei WM-Spiele und grosse Turniere hoeher gewichtet werden als normale Friendlies.

### FIFA-Rankings

FIFA-Rankings werden zeitlich korrekt eingebaut. Fuer jedes historische Spiel wird nur das Ranking verwendet, das vor diesem Spiel bekannt war.

Wichtige Ranking-Features sind:

- FIFA-Rang Heimteam
- FIFA-Rang Auswaertsteam
- Rangdifferenz
- FIFA-Punkte Heimteam
- FIFA-Punkte Auswaertsteam
- Punktedifferenz

### Optionale StatsBomb-Open-Data-Features

Wenn StatsBomb-Open-Data vorhanden ist, koennen zusaetzliche Event-Features genutzt werden:

- xG for
- xG against
- Schuesse
- Schuesse aufs Tor
- Paesse
- Pressures

Da diese Daten nicht fuer alle historischen Laenderspiele existieren, sind sie optional. Das Modell kann auch ohne diese Features laufen.

## Feature Engineering

Der Feature Builder laeuft chronologisch ueber alle Spiele. Das ist wichtig, damit kein Wissen aus der Zukunft ins Training rutscht.

Ablauf pro Spiel:

1. Features werden aus dem Zustand vor dem Spiel erzeugt.
2. Das echte Ergebnis wird erst danach verwendet, um Elo, Form, Attack/Defense-Ratings und Historien zu aktualisieren.

Dadurch entsteht eine realistische Situation: Das Modell weiss bei einem Spiel nur das, was vor Anpfiff bekannt gewesen waere.

### Wichtige Feature-Familien

#### Elo und Teamstaerke

Das Notebook baut eigene Elo-artige Ratings aus den historischen Spielen.

Beispiele:

```text
elo_home_pre
elo_away_pre
elo_diff
elo_ratio
```

WM-Spiele und grosse Turniere beeinflussen diese Ratings staerker als Friendlies.

#### Recent Form

Form wird ueber mehrere Fenster berechnet:

```text
letzte 3, 5, 10 und 20 Spiele
```

Daraus entstehen Features fuer:

- Tore geschossen
- Tore kassiert
- Tordifferenz
- Punkte
- Siegquote
- Niederlagenquote
- Clean Sheets
- Spiele ohne eigenes Tor

#### Opponent-Adjusted Form

Ein 2:0 gegen ein Topteam ist nicht dasselbe wie ein 2:0 gegen ein sehr schwaches Team. Deshalb werden Formwerte an die Gegnerstaerke angepasst.

Beispiele:

```text
home_adj_gf_l10
home_adj_ga_l10
home_adj_gd_l20
adj_form_gd_diff_l20
adj_form_pts_diff_l10
```

Diese opponent-adjusted Features waren in den Modellanalysen besonders wichtig.

#### Attack/Defense-Ratings

Neben Gesamtstaerke baut das Modell getrennte Angriffs- und Defensivsignale.

Beispiele:

```text
home_attack_pre
away_attack_pre
home_def_weak_pre
away_def_weak_pre
home_attack_vs_away_def
away_attack_vs_home_def
```

Das hilft besonders fuer genaue Score-Prognosen, weil ein Team nicht nur ?gut? oder ?schlecht? sein kann, sondern offensiv stark, defensiv stabil, offen, volatil oder drawlastig.

#### Turnier- und Venue-Kontext

Das Modell unterscheidet unter anderem:

- WM-Spiel
- Qualifikationsspiel
- Friendly
- grosses Kontinentalturnier
- neutraler Platz
- Gastgeber-/Heimvorteil

Beispiele:

```text
is_world_cup
is_qualifier
is_friendly
is_major_tournament
neutral
home_country_host
away_country_host
```

#### Head-to-Head und Shootouts

Direkte Duelle und historische Elfmeterschiessen werden ebenfalls als Zusatzsignale genutzt.

Beispiele:

```text
h2h_home_pts_l10
h2h_home_gd_l10
home_shootout_wins_pre
away_shootout_wins_pre
```

## Modellarchitektur

WeltmeisterKI4 ist kein einzelnes Modell, sondern ein kleines Ensemble-System.

### 1. XGBoost-v2

XGBoost-v2 ist die stabile Production-Basis. Es arbeitet mit den numerischen Features aus Elo, FIFA-Ranking, Form, Attack/Defense, H2H, Turnierkontext und optionalen xG-Features.

Es sagt unter anderem voraus:

- erwartete Heimtore
- erwartete Auswaertstore
- 1X2-Wahrscheinlichkeiten

Fuer Tore werden Poisson-nahe Zielsetzungen verwendet, weil Fussballtore Zaehldaten sind.

### 2. Deep-v1

Deep-v1 ist der Deep-Learning-Challenger in PyTorch. Er nutzt:

- numerische V2-Features
- Team-Embeddings
- Tournament-Embeddings
- Confederation-Embeddings
- Sequenzinformationen aus den letzten Spielen
- Multi-Task-Heads fuer Tore, 1X2, Draw, Total Goals und Goal Difference

Deep-v1 ersetzt XGBoost nicht blind. Es liefert ein zweites, andersartiges Signal.

### 3. Finales Hybridmodell

Das aktuell beste Produktionsmodell ist ein Hybrid:

```text
65% XGBoost-v2
35% Deep-v1
+ Scoreline-Kalibrierung
```

Das Hybridmodell war im Backtest besser als v2 alleine und besser als Deep alleine. Es nutzt die Stabilitaet von XGBoost und die zusaetzlichen Signale des Deep-Modells.

### 4. Scoreline-Kalibrierung

Aus den erwarteten Toren wird eine Score-Matrix gebaut:

```text
P(0:0), P(1:0), P(1:1), P(2:1), ...
```

Danach werden typische Fussballmuster kalibriert, besonders Low-Score-Ergebnisse und Remis-nahe Resultate. Dadurch werden Scores wie `0:0`, `1:0`, `0:1` und `1:1` realistischer behandelt.

## Live-Updates mit gespielten WM-Spielen

Im Inference-Notebook gibt es eine zentrale Eingabezelle:

```python
PLAYED_MATCHES_INPUT = [
    {"home": "Mexico", "away": "South Africa", "home_score": 2, "away_score": 0},
    {"home": "South Korea", "away": "Czech Republic", "home_score": 2, "away_score": 1},
]
```

Dort koennen neue echte Ergebnisse eingetragen werden. Danach werden die Prognosezellen erneut ausgefuehrt.

Die echten WM-Ergebnisse werden nicht einfach ignoriert, sondern in den aktuellen Modellzustand eingearbeitet. Ueber diesen Parameter kann man steuern, wie stark frische WM-Ergebnisse die naechsten Prognosen beeinflussen:

```python
REAL_MATCH_STATE_REPEATS = 2
```

`2` bedeutet: Das echte Ergebnis wird im State zweimal verarbeitet. Dadurch reagieren Form, Elo, Attack/Defense und Gruppenstand staerker auf aktuelle WM-Spiele, ohne das komplette Modell neu zu trainieren.

## Prognose-Outputs

Das Inference-Notebook gibt bewusst viele Zusatzstatistiken aus, damit man nicht nur blind den haeufigsten Score tippt.

### Einzelspiel-Prognosen

Pro Spiel werden ausgegeben:

- erwartete Tore / xG-nahe Modellwerte
- Sieg-Wahrscheinlichkeit
- Remis-Wahrscheinlichkeit
- Niederlage-Wahrscheinlichkeit
- wahrscheinlichster Score
- Top-8-Scorelines
- Score-Wahrscheinlichkeiten
- praktischer Tipp
- Differenz zwischen Top-1 und Top-2 Scoreline

Wichtig: `expected_score` ist kein harter Tipp, sondern der erwartete Torwert. Beispiel:

```text
expected_score = 1.74 : 0.96
```

Das bedeutet: Das Modell erwartet im Mittel etwa 1.74 Tore fuer Team A und 0.96 fuer Team B. Der wahrscheinlichste konkrete Score kann trotzdem `1:0` oder `2:1` sein.

### Gruppenphase

Das Notebook kann aus allen Gruppenpartien berechnen:

- erwartete Gruppentabellen
- Punkte
- Tore
- Gegentore
- Tordifferenz
- Gruppensieger
- Gruppenzweite
- Restprogramm unter Beruecksichtigung gespielter Spiele

Bereits gespielte Matches werden fix gesetzt. Noch offene Matches werden simuliert oder als Modellprognose ausgegeben.

### KO-Runden

Fuer KO-Spiele werden Remis-Wahrscheinlichkeiten als Gleichstand nach regulaerer Spielzeit interpretiert. Danach wird ein Sieger fuer Verlaengerung/Elfmeterschiessen simuliert.

Die Ausgaben zeigen unter anderem:

- Matchup
- erwarteter Score
- wahrscheinlichste Scoreline
- Weiterkommenswahrscheinlichkeit aus Sicht des zuerst genannten Teams
- Gleichstand-vor-nach-Elfmeterschiessen-Signal
- Gewinner

### Monte Carlo

Monte Carlo simuliert die komplette WM viele Male, z.B. 50.000 Simulationen.

Dadurch entstehen robuste Turnierwahrscheinlichkeiten:

- Wahrscheinlichkeit 16-telfinale zu erreichen
- Wahrscheinlichkeit Achtelfinale zu erreichen
- Wahrscheinlichkeit Viertelfinale zu erreichen
- Wahrscheinlichkeit Halbfinale zu erreichen
- Wahrscheinlichkeit Finale zu erreichen
- Weltmeister-Wahrscheinlichkeit
- haeufigste Exit-Runde
- haeufigster Exit-Gegner
- haeufigster Ausscheidungs-Score

Das ist fuer Tippspiele oft nuetzlicher als ein einzelner deterministischer Turnierbaum.

## Wie man Ergebnisse interpretiert

### Haeufigstes Resultat

Das haeufigste Resultat ist die einzelne Scoreline mit der hoechsten Wahrscheinlichkeit. Bei Fussball liegen die Top-Scores oft eng beieinander. Beispiel:

```text
1:0 = 13.5%
1:1 = 12.8%
```

Dann ist `1:0` zwar Top-1, aber nicht massiv sicherer als `1:1`.

### Expected Score

Der Expected Score ist der Durchschnitt der Torverteilung. Er kann z.B. `1.45:1.01` sein. Gerundet waere das `1:1`, obwohl der wahrscheinlichste einzelne Score vielleicht `1:0` ist.

### Top-8-weighted Score

Der Top-8-weighted Score betrachtet die acht wahrscheinlichsten Scorelines und bildet daraus einen gewichteten Durchschnitt. Das ist ein guter Mittelweg zwischen ?nur Top-1? und ?nur xG runden?.

### Practical Score Pick

Der praktische Tipp nimmt die Top-Scoreline, zeigt aber bei sehr knappen Top-1/Top-2-Unterschieden beide Optionen an.

Beispiel:

```text
1:0 oder 1:1
```

Das heisst: Das Modell sieht beide Ergebnisse fast gleich plausibel.

## CPU und GPU

Die Inference-Version laeuft ohne GPU.

Das Notebook setzt automatisch:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Wenn CUDA vorhanden ist, wird die GPU genutzt. Wenn nicht, wird das Modell auf CPU geladen:

```python
torch.load(..., map_location=device)
```

Fuer einzelne Spielprognosen reicht CPU locker. Monte Carlo mit sehr vielen Simulationen kann auf CPU laenger dauern, ist aber weiterhin moeglich. Wenn es zu langsam ist, kann man die Anzahl der Simulationen reduzieren:

```python
N_SIMS = 10_000
N_TREE_SIMS = 10_000
```

## Typischer Workflow

1. Notebook-Ordner in Jupyter Lab oeffnen.
2. Modellfiles in `models/` bereitstellen.
3. `WeltmeisterKI4_INFERENCE.ipynb` starten.
4. Setup- und Modellzellen ausfuehren.
5. Bereits gespielte WM-Spiele in `PLAYED_MATCHES_INPUT` eintragen.
6. Prognosezellen erneut ausfuehren.
7. Einzelspiel-Prognosen, Gruppenstaende, KO-Baum und Monte-Carlo-Outputs interpretieren.

## Kurz gesagt

WeltmeisterKI4 ist eine hybride Prognosemaschine:

```text
Historische Laenderspiele ab 1980
+ FIFA-Rankings
+ Elo/Form/Attack-Defense/H2H/xG-Features
+ XGBoost-v2
+ Deep-v1
+ Scoreline-Kalibrierung
+ Monte Carlo
= probabilistische WM-Prognosen
```

Das Modell soll nicht behaupten, die Zukunft sicher zu kennen. Es soll Wahrscheinlichkeiten transparent machen, Alternativen zeigen und bei Tippspiel-Entscheidungen eine deutlich bessere Grundlage liefern als Bauchgefuehl allein.
