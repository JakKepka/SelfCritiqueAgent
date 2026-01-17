# SelfCritiqueAgent — prototyp

Prosty prototyp pipeline'u "Agent 1 (generator) → Agent 2 (krytyk)" dla zadania prawniczego (Wariant A).

Struktura:
- `data/paremie.jsonl` — korpus paremii (mały zbiór reguł)
- `data/cases.jsonl` — prototypowe streszczenia spraw (12 przypadków z gold labels i gold paremii)
- `prompts/agent1_template.txt` — szablon promptu dla Agenta 1 (polski)
- `prompts/agent2_template.txt` — szablon promptu dla Agenta 2 (polski)
- `src/runner.py` — prosty CLI-runner prototypu (nie wykonuje wywołań LLM; daje deterministyczne odpowiedzi do testów formatu)

Szybkie uruchomienie (w repozytorium):

```bash
python3 src/runner.py list
# wypisze dostępne sprawy

python3 src/runner.py run C01
# uruchomi pipeline dla sprawy C01 (Agent1 -> Agent2, lokalny prototyp)
```

Następne kroki proponowane:
- Podpięcie Gemini/LLM: użyć plików z `prompts/` i w `src/runner.py` dodać wywołania API zamiast funkcji "naive_agent1" i "naive_agent2".
- Zaimplementować prosty retriever embeddingowy (FAISS/Pinecone) jeśli potrzeba lepszego dopasowania.
- Przygotować skrypty ewaluacyjne (accuracy, precision/recall@k, analiza błędów).
# SelfCritiqueAgent