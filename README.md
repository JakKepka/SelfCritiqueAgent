# SelfCritiqueAgent

Pipeline wieloagentowy do automatycznej analizy prawniczej z samokrytyką i oceną LLM.

## Opis projektu

System składa się z trzech agentów AI:

- **Agent 1 (Generator)**: Otrzymuje opis sprawy prawnej i listę paremii (łacińskich maksym prawnych), analizuje przypadek i wydaje orzeczenie (zasadne/niezasadne/niejednoznaczne) wraz z uzasadnieniem.
- **Agent 2 (Krytyk)**: Sprawdza rozwiązanie Agenta 1, weryfikuje poprawność zastosowania paremii, proponuje korekty i tworzy poprawione uzasadnienie sądowe.
- **Agent Judge (Ewaluator)**: Ocenia jakość uzasadnień obu agentów w porównaniu z referencyjnym uzasadnieniem (gold standard), wystawia oceny 1-5 wraz z komentarzami.

Projekt obsługuje dwa tryby działania:
- **Tryb naiwny**: Deterministyczny pipeline bez wywołań LLM (do testów)
- **Tryb LLM**: Pełna integracja z Gemini API lub OpenAI GPT

## Struktura projektu

```
├── data/
│   ├── cases.jsonl          # Zbiór 12 spraw prawniczych z gold labels
│   └── paremie.jsonl        # Korpus 20 łacińskich maksym prawnych
├── prompts/
│   ├── agent1_template.yaml # Prompt dla Agenta 1 (generator)
│   ├── agent2_template.yaml # Prompt dla Agenta 2 (krytyk)
│   └── agent_judge_template.yaml # Prompt dla LLM Judge
├── src/
│   ├── runner.py            # CLI orchestrator (main entry point)
│   ├── llm.py               # Integracja LLM (Gemini, OpenAI)
│   ├── parsers.py           # Parsowanie JSON, obliczanie metryk
│   ├── storage.py           # Zapis wyników per-case
│   └── utils.py             # Pomocnicze funkcje (wczytywanie danych)
├── scripts/
│   └── eval_aggregate.py    # Agregacja metryk, wykresy
└── results/
    └── {case_id}/           # Wyniki dla każdej sprawy
        ├── agent1_raw.txt
        ├── agent1_parsed.json
        ├── agent2_raw.txt
        ├── agent2_parsed.json
        ├── metrics_agent1.json
        ├── metrics_agent2.json
        └── metrics_decisions.json
```

## Źródła danych

### Paremie / łacińskie maksymy prawne
- [List of Latin legal terms (Wikipedia)](https://en.wikipedia.org/wiki/List_of_Latin_legal_terms)
- [Pacta sunt servanda](https://en.wikipedia.org/wiki/Pacta_sunt_servanda)
- [Res judicata (Cornell)](https://www.law.cornell.edu/wex/res_judicata)
- [Audi alteram partem](https://en.wikipedia.org/wiki/Audi_alteram_partem)
- [Ignorantia juris non excusat](https://en.wikipedia.org/wiki/Ignorantia_juris_non_excusat)
- [Nemo plus iuris](https://pl.wikipedia.org/wiki/Nemo_plus_iuris)
- [Lex superior derogat legi inferiori](https://pl.wikipedia.org/wiki/Lex_superior_derogat_legi_inferiori)

### Typowe kategorie sporów (case summaries)
- [Breach of contract (Cornell)](https://www.law.cornell.edu/wex/breach_of_contract)
- [Negligence (Cornell)](https://www.law.cornell.edu/wex/negligence)
- [Unjust enrichment (Cornell)](https://www.law.cornell.edu/wex/unjust_enrichment)
- [Promissory estoppel (Cornell)](https://www.law.cornell.edu/wex/promissory_estoppel)
- [Estoppel (Cornell)](https://www.law.cornell.edu/wex/estoppel)

## Instalacja

```bash
# Klonowanie repozytorium
git clone <repo-url>
cd SelfCritiqueAgent

# Instalacja zależności
pip install google-generativeai openai matplotlib
```

## Konfiguracja

Przed uruchomieniem w trybie LLM ustaw klucz API:

```bash
# Dla Gemini
export GEMINI_API_KEY="your-api-key"

# Lub dla OpenAI
export OPENAI_API_KEY="your-api-key"
```

Alternatywnie: utwórz plik `src/secrets.py`:
```python
GEMINI_API_KEY = "your-api-key"
OPENAI_API_KEY = "your-api-key"
```

## Użycie

### Lista dostępnych spraw
```bash
python3 src/runner.py list
```

### Uruchomienie pojedynczej sprawy

**Tryb naiwny** (bez LLM):
```bash
python3 src/runner.py run C01
```

**Tryb Gemini**:
```bash
python3 src/runner.py run C01 --use-gemini
```

**Tryb OpenAI**:
```bash
python3 src/runner.py run C01 --use-openai
```

**Z oceną uzasadnień przez LLM Judge**:
```bash
python3 src/runner.py run C01 --use-gemini --check-justification
```

### Przetwarzanie wszystkich spraw (batch)

```bash
# Tryb Gemini
python3 src/runner.py run --all --use-gemini --check-justification

# Tryb OpenAI
python3 src/runner.py run --all --use-openai --check-justification
```

### Agregacja wyników i generowanie raportów

Po przetworzeniu wszystkich spraw:
```bash
python3 scripts/eval_aggregate.py
```

Skrypt wygeneruje:
- `results/evaluation_report.json` — zagregowane metryki (precision, recall, accuracy, F1, justification scores)
- `results/evaluation_plots.png` — wykresy porównawcze (4 panele: precision/recall, decision accuracy/F1, justification scores, combined view)

## Metryki

### Agent 1 (Generator)
- **Decision match**: Zgodność decyzji z gold standard (zasadne/niezasadne/niejednoznaczne)
- **Precision/Recall (paremie)**: Poprawność doboru paremii
- **Justification score**: Heurystyczny overlap tokenów z gold justification (1-5)
- **LLM Judge score**: Ocena uzasadnienia przez LLM Judge (1-5)

### Agent 2 (Krytyk)
- **Post-correction precision/recall**: Metryki po korekcie Agenta 2
- **Recommended paremie**: Lista skorygowanych paremii
- **Uzasadnienie sądowe**: Przepisane uzasadnienie w stylu sądowym
- **LLM Judge score**: Ocena poprawionego uzasadnienia (1-5)

### Agregacja (evaluation_report.json)
- **Decision accuracy**: Odsetek poprawnych decyzji
- **Macro F1**: Średnia F1 dla klas decyzyjnych (wieloklasowa)
- **Average precision/recall**: Średnie dla doboru paremii
- **Average justification scores**: Średnie oceny uzasadnień (heuristic + LLM Judge)

## Przykładowe wyniki

```
Agent 1 metrics:
  Decision accuracy: 0.83
  Macro F1: 0.79
  Average precision: 0.74
  Average recall: 0.68
  Average justification score (heuristic): 3.2
  Average LLM Judge score: 3.8

Agent 2 metrics:
  Average precision (post-correction): 0.81
  Average recall (post-correction): 0.75
  Average LLM Judge score: 4.1
```

## Licencja

Zobacz plik [LICENSE](LICENSE).