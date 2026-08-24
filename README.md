# ProjektDataAnalysis

Python-basierte Analyse von Amazon Fine Food Reviews mittels Natural Language Processing (NLP).

## Projektbeschreibung

Dieses Projekt wurde im Rahmen des IU-Moduls **Data Analysis** entwickelt. Ziel ist die Analyse einer Sammlung von Amazon Fine Food Reviews, um häufig diskutierte Themen automatisch zu identifizieren.

Der Datensatz besteht aus Produktbewertungen in englischer Sprache. Die Texte werden zunächst vorverarbeitet und anschließend mithilfe verschiedener Verfahren zur Vektorisierung und Themenmodellierung analysiert.

Zur Bestimmung einer geeigneten Themenanzahl wird zusätzlich der Coherence Score verwendet. Die erzeugten Themenmodelle werden anschließend anhand weiterer Kennzahlen miteinander verglichen.

---

## Verwendete Technologien

- Python 3.13
- pandas
- NumPy
- NLTK
- scikit-learn
- Gensim
- matplotlib

---

## Verwendete Verfahren

### Textvorverarbeitung

Vor der eigentlichen Analyse werden die Texte bereinigt.

Dabei werden unter anderem folgende Verarbeitungsschritte durchgeführt:

- Umwandlung in Kleinbuchstaben
- Entfernung von HTML-Tags
- Entfernung von Zahlen
- Entfernung von Sonderzeichen und Satzzeichen
- Entfernung englischer Stopwörter

---

### Vektorisierung

Die bereinigten Texte werden anschließend in numerische Merkmalsvektoren umgewandelt.

Verwendete Verfahren:

- Bag of Words (BoW)
- Term Frequency–Inverse Document Frequency (TF-IDF)

Für die anschließende Themenmodellierung wird die Bag-of-Words-Darstellung für LDA und die TF-IDF-Darstellung für NMF verwendet.

---

### Themenmodellierung

Zur automatischen Identifikation häufig diskutierter Themen werden zwei Verfahren eingesetzt:

- Latent Dirichlet Allocation (LDA)
- Non-negative Matrix Factorization (NMF)

Die Themen werden anhand ihrer jeweils wichtigsten Begriffe beschrieben.

---

### Evaluation

Zur Bestimmung einer geeigneten Themenanzahl wird für unterschiedliche Anzahlen von Topics der **Coherence Score (C_v)** berechnet.

Die resultierenden Themenmodelle werden zusätzlich anhand folgender Kennzahlen miteinander verglichen:

- Topic Diversity
- durchschnittliche Topic-Überschneidung

Ergänzend erfolgt eine qualitative Betrachtung der extrahierten Themen anhand ihrer wichtigsten Keywords.

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
│   ├── evaluation.py
│   ├── main.py
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

### Repository klonen

```bash
git clone https://github.com/mikesauer8/ProjektDataAnalysis.git
```

### In das Projektverzeichnis wechseln

```bash
cd ProjektDataAnalysis
```

### Virtuelle Umgebung erstellen

```bash
python -m venv .venv
```

### Virtuelle Umgebung aktivieren

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

---

## Projekt ausführen

Das vollständige Analyseprogramm wird über `main.py` gestartet.

Windows:

```bash
python .\src\main.py
```

Linux/macOS:

```bash
python ./src/main.py
```

Während der Ausführung werden die Daten geladen und vorverarbeitet, die Vektordarstellungen erzeugt, die Themenmodelle trainiert und die Evaluationskennzahlen berechnet.

---

## Erzeugte Dateien

Nach der Ausführung werden die Ergebnisse automatisch in den Verzeichnissen `results/` und `figures/` gespeichert.

### Ergebnisdateien

```text
results/
├── coherence_scores.txt
├── evaluation.txt
├── lda_topics.txt
├── nmf_topics.txt
└── topic_overview.txt
```

Die Dateien enthalten die berechneten Coherence Scores, die Evaluationskennzahlen, die extrahierten Topics sowie eine Übersicht der identifizierten Themen.

### Visualisierungen

```text
figures/
├── coherence_scores.png
├── model_evaluation.png
└── top_words_bow.png
```

Die Abbildungen visualisieren die Coherence Scores in Abhängigkeit von der Themenanzahl, den Vergleich der Topic-Modelle sowie die häufigsten Wörter der Bag-of-Words-Darstellung.

---

## Datensatz

Verwendeter Datensatz:

**Amazon Fine Food Reviews**

https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

Aus Lizenz- und Größengründen befindet sich der Datensatz nicht im Repository und muss separat heruntergeladen werden.

Nach dem Download muss die Datei

```text
Reviews.csv
```

im Verzeichnis

```text
data/
```

abgelegt werden.

Für die Analyse wird aus dem Datensatz eine reproduzierbare Stichprobe von **5.000 Bewertungen** verwendet.

---

## Projektstand

- ✔ Datensatz eingebunden
- ✔ Textvorverarbeitung implementiert
- ✔ Bag of Words implementiert
- ✔ TF-IDF implementiert
- ✔ LDA implementiert
- ✔ NMF implementiert
- ✔ Coherence Score implementiert
- ✔ Automatische Bestimmung einer geeigneten Themenanzahl
- ✔ Evaluation der Topic-Modelle
- ✔ Übersicht der identifizierten Themen und Keywords
- ✔ Automatische Speicherung der Ergebnisse
- ✔ Erstellung von Visualisierungen
- ✔ Projektdokumentation

---

## Autor

Mike Sauer  
IU Internationale Hochschule  
Modul: Data Analysis