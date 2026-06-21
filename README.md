# WeltmeisterKI2026

WeltmeisterKI4 ist ein datengetriebenes Prognose-System fuer die FIFA WM 2026. Es kombiniert historische Laenderspiele, FIFA-Rankings, Elo-/Form-Features, XGBoost, ein kleines PyTorch-Deep-Learning-Modell und Monte-Carlo-Simulationen.

Das Ziel ist nicht nur ein einzelner Tipp wie `2:1`, sondern eine komplette Wahrscheinlichkeitsanalyse:

- Sieg / Remis / Niederlage
- erwartete Tore
- wahrscheinlichste Scorelines
- Top-8-Scorelines
- Gruppenstaende
- KO-Pfade
- Titelchancen
- Ausscheidungswahrscheinlichkeiten
- Word-Report-Export

## Schnellstart

1. Repository herunterladen oder klonen.
2. Modellartefakte aus Google Drive herunterladen:

   [Modelle und Zusatzdateien auf Google Drive](https://drive.google.com/drive/folders/1rjJ2NC4eMc7OTblni3bcv1Wq5PkOYfxn?usp=sharing)

3. Den heruntergeladenen `models/`-Ordner in denselben Ordner legen wie das Notebook.
4. Jupyter Lab in diesem Projektordner starten.
5. `WeltmeisterKI4_INFERENCE.ipynb` oeffnen.
6. Zellen von oben nach unten ausfuehren.
7. Neue WM-Ergebnisse in der 72-Spiele-Zelle eintragen.
8. Danach die Prognose-, Monte-Carlo- und Exportzellen erneut ausfuehren.

## Ordnerstruktur

So sollte der Projektordner aussehen:

```text
WeltmeisterKI4/
├─ README.md
├─ WeltmeisterKI4_INFERENCE.ipynb
├─ WeltmeisterKI4_SOTA_Final.ipynb
├─ teams.csv
├─ matches.csv
├─ tournament_stages.csv
├─ host_cities.csv
└─ models/
   ├─ xgb_goal_models_v2_ensemble.joblib
   ├─ weltmeisterki_deep_v1.pt
   └─ weltmeisterki4_sota_meta.joblib
```

Der `data/`-Ordner muss nicht hochgeladen werden. Er wird beim Ausfuehren des Notebooks neu erzeugt und befuellt. Darin landen heruntergeladene Rohdaten, erzeugte Features und Cache-Dateien.

## Notebook-Varianten

### `WeltmeisterKI4_INFERENCE.ipynb`

Das ist die Nutzer-Version.

Sie:

- trainiert kein neues Modell
- laedt die fertigen Modellartefakte aus `models/`
- funktioniert auch ohne GPU
- aktualisiert Prognosen mit bereits gespielten WM-Spielen
- gibt Zusatzstatistiken fuer Tippspiel-Entscheidungen aus
- simuliert Gruppenphase, KO-Runden und komplette Turnierbaeume
- kann einen Word-Report exportieren

Diese Version ist die richtige Wahl, wenn man einfach Prognosen erzeugen moechte.

### `WeltmeisterKI4_SOTA_Final.ipynb`

Das ist die Research- und Trainingsversion.

Sie:

- baut Features
- trainiert XGBoost-Modelle
- trainiert das Deep-Learning-Modell
- rechnet Backtests
- erzeugt die Modellartefakte fuer `models/`

Diese Version ist rechenintensiver und eher fuer Entwicklung, Tuning und Modellvergleich gedacht.

## Modellartefakte

Die Inference-Version braucht diese Dateien:

```text
models/xgb_goal_models_v2_ensemble.joblib
models/weltmeisterki_deep_v1.pt
models/weltmeisterki4_sota_meta.joblib
```

Download:

[Google Drive Modellordner](https://drive.google.com/drive/folders/1rjJ2NC4eMc7OTblni3bcv1Wq5PkOYfxn?usp=sharing)

Das Notebook sucht zuerst in:

```text
PROJECT/models/
```

Wenn Jupyter direkt im Projektordner gestartet wird, passt `PROJECT = Path.cwd().resolve()` automatisch.

## Daten

Das Notebook arbeitet mit kostenlosen bzw. offenen Datenquellen.

### Historische Laenderspiele

Die historische Basis kommt aus internationalen Spielresultaten ab 1980. Genutzt werden unter anderem:

- Datum
- Heimteam
- Auswaertsteam
- Tore
- Turnier
- neutraler Platz
- Torschuetzen
- Elfmeterinformationen
- Eigentore
- Shootouts

Das Modell wird nicht nur auf WM-Spielen trainiert, weil das viel zu wenige Datenpunkte waeren. Stattdessen nutzt es alle Laenderspiele ab 1980 und gewichtet WM- und Turnierspiele staerker als Friendlies.

### FIFA-Rankings

FIFA-Rankings werden zeitlich korrekt eingebaut. Fuer jedes historische Spiel wird nur das Ranking verwendet, das vor diesem Spiel bekannt war.

Wichtige Ranking-Features:

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

Das Modell laeuft auch ohne vollstaendige StatsBomb-Daten weiter.

## Feature Engineering

Der Feature Builder laeuft chronologisch ueber alle historischen Spiele.

Ablauf:

1. Fuer ein Spiel werden nur Informationen verwendet, die vor diesem Spiel bekannt waren.
2. Danach wird das echte Ergebnis benutzt, um Elo, Form, Attack/Defense-Ratings und Historien zu aktualisieren.

Dadurch wird Data Leakage vermieden.

Wichtige Feature-Familien:

- Elo und Teamstaerke
- FIFA-Ranking
- Recent Form ueber 3, 5, 10 und 20 Spiele
- opponent-adjusted Form
- Attack-/Defense-Ratings
- Head-to-Head
- Turnierkontext
- neutraler Platz / Gastgebervorteil
- Shootout-Erfahrung
- optionale xG-/Event-Features

## Modellarchitektur

WeltmeisterKI4 ist kein einzelnes Modell, sondern ein kleines Ensemble-System.

### 1. XGBoost-v2

XGBoost-v2 ist die stabile tabellarische Basis. Es nutzt Elo, FIFA-Ranking, Form, Attack/Defense, Head-to-Head, Turnierkontext und optionale xG-Features.

Es sagt unter anderem vorher:

- erwartete Heimtore
- erwartete Auswaertstore
- 1X2-Wahrscheinlichkeiten

### 2. Deep-v1

Deep-v1 ist ein PyTorch-Challenger mit:

- numerischen V2-Features
- Team-Embeddings
- Tournament-Embeddings
- Confederation-Embeddings
- Sequenzinformationen aus den letzten Spielen
- Multi-Task-Heads fuer Tore, 1X2, Draw, Total Goals und Goal Difference

### 3. Finales Hybridmodell

Das finale Produktionsmodell kombiniert beide Welten:

```text
XGBoost-v2
+ Deep-v1
+ Scoreline-Kalibrierung
= finales Hybridmodell
```

Das Hybridmodell nutzt die Stabilitaet von XGBoost und die zusaetzlichen Signale des Deep-Modells.

### 4. Scoreline-Kalibrierung

Aus erwarteten Toren wird eine Score-Matrix gebaut:

```text
P(0:0), P(1:0), P(1:1), P(2:1), ...
```

Danach werden typische Fussballmuster kalibriert, besonders Low-Score-Ergebnisse und Remis-nahe Resultate.

## Live-Updates mit echten WM-Ergebnissen

Im Inference-Notebook gibt es eine zentrale Zelle mit allen 72 Gruppenspielen.

Bereits bekannte Ergebnisse stehen als Zahlen drin. Offene Spiele haben:

```python
"home_score": None
"away_score": None
```

Wenn ein neues Ergebnis dazukommt, nur diese beiden Werte ersetzen und danach die folgenden Prognosezellen erneut ausfuehren.

Der Live-Faktor steht auf:

```python
REAL_MATCH_STATE_REPEATS = 2
```

Das bedeutet: Echte WM-Ergebnisse werden doppelt in den aktuellen Forecast-State geschrieben. Dadurch reagieren Elo, Form, Attack/Defense und Gruppenstand staerker auf aktuelle Turnierform.

## Prognose-Outputs

Das Notebook erzeugt bewusst viele Zusatzstatistiken.

### Einzelspiel-Prognosen

Pro Spiel:

- erwartete Tore
- Sieg-Wahrscheinlichkeit
- Remis-Wahrscheinlichkeit
- Niederlage-Wahrscheinlichkeit
- wahrscheinlichster Score
- Top-8-Scorelines
- Score-Wahrscheinlichkeiten
- praktischer Tipp
- Abstand zwischen Top-1 und Top-2 Scoreline

### Gruppenphase

Das Notebook berechnet:

- Gruppentabellen
- Punkte
- Tore
- Gegentore
- Tordifferenz
- Gruppensieger
- Gruppenzweite
- Restprogramm unter Beruecksichtigung gespielter Spiele

### KO-Runden

Fuer KO-Spiele wird ein Remis als Gleichstand nach regulaerer Spielzeit interpretiert. Danach wird ein Sieger fuer Verlaengerung/Elfmeterschiessen simuliert.

### Monte Carlo

Monte Carlo simuliert die komplette WM viele Male, z.B. 50.000 Simulationen.

Dadurch entstehen:

- 16telfinale-Wahrscheinlichkeit
- Achtelfinale-Wahrscheinlichkeit
- Viertelfinale-Wahrscheinlichkeit
- Halbfinale-Wahrscheinlichkeit
- Finalwahrscheinlichkeit
- Weltmeister-Wahrscheinlichkeit
- haeufigste Exit-Runde
- haeufigster Exit-Gegner
- haeufigster Ausscheidungs-Score

## Word-Report

Am Ende kann ein Word-Report erzeugt werden. Der Report enthaelt:

- Setup
- Einzel-Durchlauf
- Gruppenspiele
- KO-Spiele
- aggregierten Monte-Carlo-Turnierbaum
- Titelwahrscheinlichkeiten
- Medaillenwahrscheinlichkeiten
- Team-spezifische Rundenverteilungen
- Ausscheidungsanalyse

Der Report wird im Projektordner gespeichert.

## CPU und GPU

Die Inference-Version funktioniert ohne GPU.

Das Notebook waehlt automatisch:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Wenn CUDA vorhanden ist, wird die GPU genutzt. Wenn nicht, laeuft das Modell auf CPU. Einzelspiel-Prognosen sind auf CPU kein Problem. Monte Carlo mit vielen Simulationen kann laenger dauern.

Falls es zu langsam ist:

```python
N_SIMS = 10_000
N_TREE_SIMS = 10_000
```

## Installation

Empfohlen ist ein Conda- oder venv-Environment mit Python 3.10 oder 3.11.

Wichtige Pakete:

```text
numpy
polars
pyarrow
scikit-learn
xgboost
torch
joblib
python-docx
```

Beispiel:

```bash
pip install numpy polars pyarrow scikit-learn xgboost torch joblib python-docx
```

## Typischer Workflow

1. Repo herunterladen.
2. Modellordner aus Google Drive herunterladen.
3. `models/` in den Projektordner legen.
4. Jupyter Lab im Projektordner starten.
5. `WeltmeisterKI4_INFERENCE.ipynb` oeffnen.
6. Zellen von oben nach unten ausfuehren.
7. Neue Ergebnisse in der 72-Spiele-Zelle eintragen.
8. Prognosezellen erneut ausfuehren.
9. Optional Word-Report exportieren.

## Interpretation

### Haeufigstes Resultat

Das haeufigste Resultat ist die einzelne Scoreline mit der hoechsten Wahrscheinlichkeit. Bei Fussball liegen Top-Scores oft eng beieinander.

Beispiel:

```text
1:0 = 13.5%
1:1 = 12.8%
```

Dann ist `1:0` zwar Top-1, aber nicht viel sicherer als `1:1`.

### Expected Score

Der Expected Score ist der Durchschnitt der Torverteilung.

Beispiel:

```text
expected_score = 1.74 : 0.96
```

Das ist kein harter Tipp, sondern der durchschnittliche Torwert des Modells.

### Practical Score Pick

Der Practical Score Pick ist fuer Tippspiele gedacht. Wenn die wahrscheinlichsten Scores sehr nah beieinander liegen, zeigt er mehrere plausible Tipps.

## Kurz gesagt

```text
Historische Laenderspiele ab 1980
+ FIFA-Rankings
+ Elo/Form/Attack-Defense/H2H/xG-Features
+ XGBoost-v2
+ Deep-v1
+ Scoreline-Kalibrierung
+ Monte Carlo
= WeltmeisterKI4
```

WeltmeisterKI4 soll nicht behaupten, die Zukunft sicher zu kennen. Es macht Wahrscheinlichkeiten sichtbar und hilft, Tipps und Turnierpfade datengetrieben einzuschaetzen.
