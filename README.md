# ProjektDataAnalysis

Python-basierte Analyse von Amazon Fine Food Reviews mittels Natural Language Processing (NLP).

## Projektbeschreibung

Dieses Projekt wurde im Rahmen des IU-Moduls **Data Analysis** entwickelt. Ziel ist die Analyse einer Sammlung von Amazon Fine Food Reviews, um häufig diskutierte Themen automatisch zu identifizieren. Hierzu werden verschiedene Verfahren der Textvorverarbeitung, Vektorisierung und Themenmodellierung eingesetzt.

Der Datensatz besteht aus Produktbewertungen in englischer Sprache und wird zunächst bereinigt und anschließend mit Methoden des Natural Language Processing (NLP) analysiert.

---

## Verwendete Technologien

- Python 3.13
- pandas
- NumPy
- NLTK
- scikit-learn
- matplotlib

---

## Verwendete Verfahren

### Textvorverarbeitung

Vor der eigentlichen Analyse werden die Texte bereinigt.

Dabei werden unter anderem

- Umwandlung in Kleinbuchstaben
- Entfernung von HTML-Tags
- Entfernung von Zahlen
- Entfernung von Sonderzeichen
- Entfernung englischer Stopwörter

durchgeführt.

---

### Vektorisierung

Die bereinigten Texte werden anschließend in numerische Merkmalsvektoren umgewandelt.

Verwendete Verfahren:

- Bag of Words (BoW)
- Term Frequency–Inverse Document Frequency (TF-IDF)

---

### Themenmodellierung

Zur Identifikation häufig auftretender Themen werden zwei Verfahren eingesetzt.

- Latent Dirichlet Allocation (LDA)
- Non-negative Matrix Factorization (NMF)

---

### Evaluation

Die erzeugten Themenmodelle werden anhand zweier Kennzahlen miteinander verglichen.

- Topic Diversity
- Durchschnittliche Topic-Überschneidung

Zusätzlich erfolgt eine qualitative Betrachtung der extrahierten Themen.

---

## Projektstruktur

```text
ProjektDataAnalysis/
│
├── data/
│   └── Reviews.csv
│
├── figures/
│
├── results/
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

Repository klonen

```bash
git clone https://github.com/<username>/ProjektDataAnalysis.git
```

In das Projektverzeichnis wechseln

```bash
cd ProjektDataAnalysis
```

Virtuelle Umgebung erstellen

```bash
python -m venv .venv
```

Virtuelle Umgebung aktivieren

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

---

## Projekt ausführen

```bash
python .\src\main.py
```

Nach der Ausführung werden automatisch folgende Dateien erzeugt:

```text
ProjektDataAnalysis/
│
├── results/
│   ├── lda_topics.txt
│   ├── nmf_topics.txt
│   └── evaluation.txt
│
└── figures/
    ├── top_words_bow.png
    └── model_evaluation.png
```

---

## Datensatz

Verwendeter Datensatz:

**Amazon Fine Food Reviews**

https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

Aus Lizenz- und Größen­gründen befindet sich der Datensatz nicht im Repository und muss separat heruntergeladen werden.

Nach dem Download ist die Datei

```text
Reviews.csv
```

im Ordner

```text
data/
```

abzulegen.

---

## Aktueller Projektstand

- ✔ Datensatz eingebunden
- ✔ Textvorverarbeitung implementiert
- ✔ Bag of Words implementiert
- ✔ TF-IDF implementiert
- ✔ LDA implementiert
- ✔ NMF implementiert
- ✔ Evaluation der Topic-Modelle
- ✔ Automatische Speicherung der Ergebnisse
- ✔ Erstellung von Visualisierungen
- ✔ Projektdokumentation

---

## Autor

Mike Sauer

IU Internationale Hochschule

Modul: Data Analysis

