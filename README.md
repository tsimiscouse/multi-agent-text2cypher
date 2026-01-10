# Multi-Agent Text-to-Cypher

Penelitian evaluasi sistem multi-agent untuk meningkatkan akurasi Text-to-Cypher pada Knowledge Graph kurikulum berbahasa Indonesia.

## Deskripsi

Penelitian ini mengimplementasikan dan mengevaluasi sistem multi-agent iterative refinement untuk tugas Text-to-Cypher, dengan membandingkan performanya terhadap baseline single-shot generation (kg-axel).

**Baseline:** kg-axel (CoT + only_paths)
**Proposed:** Multi-Agent Iterative Refinement System with ReAct (Reasoning and Acting)

### Fitur Utama
- **ReAct Always-On**: Seluruh sistem menggunakan ReAct prompting untuk transparansi penalaran
- **6 Agents**: 4 LLM agents dengan ReAct + 2 rule-based agents
- **Comprehensive Metrics**: 23+ metrik evaluasi mengikuti format kg-axel baseline
- **Iterative Refinement**: Maksimal k=3 iterasi untuk perbaikan query

## Struktur Project

```
multi-agent-text2cypher/
├── agents/                  # Multi-agent components
│   ├── query_generator.py   # Agent 1: Query generation & refinement
│   ├── query_evaluator.py   # Agent 2: LLM-based query evaluation
│   ├── entity_extractor.py  # Agent 3: Named entity extraction
│   ├── entity_verifier.py   # Agent 4: Entity verification & correction
│   ├── instructions_generator.py  # Agent 5: Correction instructions
│   └── feedback_aggregator.py     # Agent 6: Feedback synthesis
│
├── baseline/                # Baseline system (kg-axel)
│   ├── kg_axel_generator.py
│   └── prompts.py
│
├── system/                  # Multi-agent orchestration
│   ├── orchestrator.py      # Main multi-agent coordinator
│   ├── graph_executor.py    # Neo4j query executor
│   └── agent_state.py       # State management
│
├── notebooks/               # Experiment notebooks
│   ├── 01_inference_baseline.ipynb
│   ├── 02_inference_multiagent.ipynb
│   ├── 03_metrics_evaluation.ipynb
│   └── 04_efficiency_analysis.ipynb
│
├── data/                    # Dataset
│   ├── ground_truth/
│   │   ├── ground_truth_52.csv     # Original from kg-axel
│   │   └── ground_truth_100.csv    # Expanded dataset
│   └── schemas/
│       ├── only_paths.txt
│       ├── nodes_and_paths.txt
│       └── full_schema.txt
│
├── prompts/                 # Prompt templates
│   ├── templates/
│   │   ├── query_generator.txt
│   │   ├── query_evaluator.txt
│   │   ├── entity_verifier.txt
│   │   └── feedback_aggregator.txt
│   └── examples/
│
├── output/                  # Experiment outputs
│   ├── baseline_kg_axel.csv
│   ├── multiagent_k3.csv
│   └── multiagent_k3_states.json
│
├── outputMetrics/          # Evaluation metrics
│   ├── comparison_metrics.csv
│   ├── ablation_study.csv
│   └── cost_benefit_analysis.csv
│
├── evaluation/             # Evaluation modules
│   ├── metrics.py
│   ├── comparator.py
│   └── analyzer.py
│
├── config/                 # Configuration files
│   ├── model_config.yaml
│   └── experiment_config.yaml
│
├── docs/                   # Documentation
│   ├── architecture.md
│   ├── agent_specifications.md
│   └── experiment_protocol.md
│
└── utils/                  # Utility functions
    ├── prompt_loader.py
    ├── schema_loader.py
    └── logger.py
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys and Neo4j credentials
```

## Quick Start

### 1. Run Baseline Experiment (kg-axel)

```bash
jupyter notebook notebooks/01_inference_baseline.ipynb
```

### 2. Run Multi-Agent Experiment

```bash
jupyter notebook notebooks/02_inference_multiagent.ipynb
```

### 3. Evaluate Metrics

```bash
jupyter notebook notebooks/03_metrics_evaluation.ipynb
```

### 4. Analyze Efficiency

```bash
jupyter notebook notebooks/04_efficiency_analysis.ipynb
```

## Research Questions

**RQ1:** Seberapa efektif multi-agent refinement dalam meningkatkan Pass@k dibandingkan single-shot generation (kg-axel baseline)?

**RQ2:** Pada jenis query apa multi-agent memberikan improvement terbesar?

**RQ3:** Apa trade-off antara peningkatan akurasi dengan computational cost?

## Metrics

### Comprehensive Evaluation Metrics (23+ kolom)

**Cypher Text Similarity:**
- BLEU, Rouge-L (Precision/Recall/F1-score), Jaro Similarity, Jaccard Similarity

**Output Similarity:**
- Jaccard Output (eksekusi output similarity)
- Pass@1 Output (output match sempurna)

**Validators:**
- Syntax Validator (Cypher syntax validation)
- Schema Validator (KG schema compliance)
- Properties Validator (property name validation)

**Derived Scores:**
- Pass@1 Score (0 atau 100)
- KG Validity Score (gabungan 3 validators: 0 atau 100)
- Jaccard Output Score (0-100)
- JaRou Score (rata-rata Jaro + Rouge-L)

**Composite Metric:**
- **LLMetric-Q**: Skor komposit (0-100) dengan bobot:
  - 30% Pass@1 Score
  - 40% KG Validity Score
  - 20% Jaccard Output Score
  - 10% JaRou Score

**ReAct-Specific Metrics:**
- Total Reasoning Steps (jumlah langkah penalaran)
- Avg Reasoning Quality (kualitas penalaran 0-1)
- Total Iterations (jumlah iterasi refinement)

## Experiment Configuration

```yaml
Baseline:
  System: kg-axel
  Technique: CoT (Chain-of-Thought)
  Schema: only_paths
  Max k: 1 (single-shot)
  Output: outputMetrics/baseline_kg_axel.csv

Multi-Agent:
  System: Multi-Agent Iterative Refinement
  Technique: ReAct (Reasoning and Acting) - Always Enabled
  Max k: 3
  Agents: 6 total
    - LLM (with ReAct): Generator, Evaluator, Instructions, Aggregator
    - Rule-based: Entity Extractor, Entity Verifier
  Output: outputMetrics/multiagent_react.csv + states JSON

Dataset:
  Questions: 52 (from kg-axel)
  Complexity: Easy, Medium, Hard
  Reasoning Level: Fakta Eksplisit, Fakta Implisit
  Sublevel: Nodes, One-hop, Multi-hop

Model:
  Default: qwen/qwen-2.5-coder-32b-instruct
  Temperature: 0.0
  Max Tokens: 512 (baseline), 1024 (multi-agent with ReAct)
```

## Results

Results tersimpan di folder `outputMetrics/` dengan format CSV:

### 1. Baseline (kg-axel)
File: `outputMetrics/baseline_kg_axel.csv`
- 52 baris (1 per pertanyaan)
- 23+ kolom metrik komprehensif
- Teknik Prompt: "CoT"

### 2. Multi-Agent (ReAct)
File: `outputMetrics/multiagent_react.csv`
- 52 baris (1 per pertanyaan)
- 23+ kolom metrik komprehensif + metrik ReAct tambahan
- Teknik Prompt: "ReAct Multi-Agent"

### 3. State Traces
File: `outputMetrics/multiagent_react_states.json`
- Detail reasoning traces untuk setiap iterasi
- Thought/Action/Observation sequences
- Quality scores per agent

**Kolom Output (identik untuk baseline & multi-agent):**
- ID Pertanyaan, Teknik Prompt Engineering, Format Representasi Skema KG
- Tingkat Penalaran, Sublevel, Tingkat Kompleksitas
- Cypher LLM, Syntax Validator, Schema Validator, Properties Validator
- Pass@1 Output, Jaccard Output
- BLEU, Rouge-L (Precision/Recall/F1), Jaro Similarity, Jaccard Similarity
- Pass@1 Score, KG Validity Score, Jaccard Output Score, JaRou Score, LLMetric-Q
- total_tokens, input_tokens, output_tokens

## Citation

```bibtex
@thesis{multiagent2025,
  title={Multi-Agent Iterative Refinement untuk Text-to-Cypher pada Knowledge Graph Kurikulum},
  author={[Your Name]},
  year={2025},
  school={[Your University]}
}
```

## License

MIT License

## Contact

[Your Contact Information]
