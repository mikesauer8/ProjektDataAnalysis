# ProjektDataAnalysis

Python-basierte Analyse von Amazon Fine Food Reviews mittels Natural Language Processing (NLP).

## Projektbeschreibung

Dieses Projekt wurde im Rahmen des IU-Moduls **Data Analysis** entwickelt. Ziel ist die Analyse einer Sammlung von Amazon Fine Food Reviews, um häufig diskutierte Themen automatisch zu identifizieren. Hierzu werden verschiedene Verfahren der Textvorverarbeitung, Vektorisierung und Themenmodellierung eingesetzt.

Der Datensatz besteht aus Produktbewertungen in englischer Sprache und wird zunächst bereinigt und anschließend mit Methoden des Natural Language Processing (NLP) analysiert. Für die Themenmodellierung werden Latent Dirichlet Allocation (LDA) und Non-negative Matrix Factorization (NMF) miteinander verglichen.

---

## Verwendete Technologien

- Python 3.13
- pandas
- NumPy
- NLTK
- scikit-learn
- gensim
- matplotlib

---

## Verwendete Verfahren

### Textvorverarbeitung

Vor der eigentlichen Analyse werden die Texte bereinigt.

Dabei werden unter anderem folgende Schritte durchgeführt:

- Umwandlung in Kleinbuchstaben
- Entfernung von HTML-Tags
- Entfernung von Zahlen
- Entfernung von Sonderzeichen
- Entfernung englischer Stopwörter

---

### Vektorisierung

Die bereinigten Texte werden anschließend in numerische Merkmalsvektoren umgewandelt.

Verwendete Verfahren:

- Bag of Words (BoW)
- Term Frequency–Inverse Document Frequency (TF-IDF)

---

### Themenmodellierung

Zur Identifikation häufig auftretender Themen werden zwei Verfahren eingesetzt:

- Latent Dirichlet Allocation (LDA) auf Grundlage der Bag-of-Words-Darstellung
- Non-negative Matrix Factorization (NMF) auf Grundlage der TF-IDF-Darstellung

Die Anzahl der Topics wird nicht ausschließlich manuell festgelegt. Für Themenanzahlen von 2 bis 10 wird für beide Verfahren der Coherence Score (C_v) berechnet. Anschließend wird für jedes Modell die Themenanzahl mit dem höchsten Coherence Score ausgewählt.

Für die verwendete reproduzierbare Stichprobe von 5.000 Bewertungen ergaben sich:

- LDA: 10 Topics, C_v = 0,504
- NMF: 9 Topics, C_v = 0,569

---

### Evaluation

Die erzeugten Themenmodelle werden anhand mehrerer Kennzahlen miteinander verglichen:

- Coherence Score (C_v)
- Topic Diversity
- Durchschnittliche Topic-Überschneidung

Für die finalen Modelle ergaben sich folgende Werte:

| Modell | Topics | Coherence Score (C_v) | Topic Diversity | Topic-Überschneidung |
|---|---:|---:|---:|---:|
| LDA | 10 | 0,504 | 0,740 | 0,070 |
| NMF | 9 | 0,569 | 0,933 | 0,009 |

NMF erzielt damit sowohl einen höheren maximalen Coherence Score als auch eine höhere Topic Diversity und eine geringere Überschneidung zwischen den identifizierten Themen.

Zusätzlich werden die extrahierten Themen und ihre wichtigsten Keywords ausgegeben und für die weitere Auswertung gespeichert.

---

## Projektstruktur

```text
ProjektDataAnalysis/
│
├── data/
│   └── Reviews.csv
│
├── figures/
│   ├── coherence_scores.png
│   ├── model_evaluation.png
│   └── top_words_bow.png
│
├── results/
│   ├── coherence_scores.txt
│   ├── evaluation.txt
│   ├── lda_topics.txt
│   ├── nmf_topics.txt
│   └── topic_overview.txt
│
├── src/
│   ├── main.py
│   ├── evaluation.py
│   ├── preprocessing.py
│   ├── topic_modeling.py
│   ├── vectorization.py
│   └── visualization.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

Repository klonen:

```bash
git clone https://github.com/mikesauer8/ProjektDataAnalysis.git
```

In das Projektverzeichnis wechseln:

```bash
cd ProjektDataAnalysis
```

Virtuelle Umgebung erstellen:

```bash
python -m venv .venv
```

Virtuelle Umgebung aktivieren:

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

---

## Projekt ausführen

Unter Windows:

```bash
python .\src\main.py
```

Alternativ unter Linux/macOS:

```bash
python ./src/main.py
```

Während der Ausführung werden die Texte vorverarbeitet, BoW- und TF-IDF-Darstellungen erzeugt, die Coherence Scores für unterschiedliche Themenanzahlen berechnet und anschließend die finalen LDA- und NMF-Modelle trainiert und evaluiert.

Nach der Ausführung werden unter anderem folgende Dateien automatisch erzeugt:

```text
results/
├── coherence_scores.txt
├── evaluation.txt
├── lda_topics.txt
├── nmf_topics.txt
└── topic_overview.txt

figures/
├── coherence_scores.png
├── model_evaluation.png
└── top_words_bow.png
```

---

## Datensatz

Verwendeter Datensatz:

**Amazon Fine Food Reviews**

https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

Aus Lizenz- und Größengründen befindet sich der Datensatz nicht im Repository und muss separat heruntergeladen werden.

Nach dem Download ist die Datei

```text
Reviews.csv
```

im Ordner

```text
data/
```

abzulegen.

Für die Analyse wird eine reproduzierbare Stichprobe von 5.000 Bewertungen mit einem festen `random_state` verwendet.

---

## Aktueller Projektstand

- ✔ Datensatz eingebunden
- ✔ Reproduzierbare Stichprobe implementiert
- ✔ Textvorverarbeitung implementiert
- ✔ Bag of Words implementiert
- ✔ TF-IDF implementiert
- ✔ LDA implementiert
- ✔ NMF implementiert
- ✔ Coherence Score (C_v) implementiert
- ✔ Vergleich verschiedener Themenanzahlen implementiert
- ✔ Automatische Auswahl der Themenanzahl
- ✔ Topic Diversity berechnet
- ✔ Topic-Überschneidung berechnet
- ✔ Themenübersicht mit Keywords erzeugt
- ✔ Automatische Speicherung der Ergebnisse
- ✔ Erstellung von Visualisierungen
- ✔ Projektdokumentation

---

## Autor

Mike Sauer

IU Internationale Hochschule  
Modul: Data Analysis

