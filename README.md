# Do Biases Stack?  
## Investigating Compound Bias Interactions in LLM-as-a-Judge Evaluation

---

## 📌 Project Overview
This project explores a critical and under-researched question in modern Natural Language Processing (NLP):

> **Do multiple biases in LLM-based evaluation systems interact, and if so, how?**

Large Language Models (LLMs) such as GPT-4 are increasingly used as automated judges for evaluating AI-generated content. While individual biases in these systems are well-documented, existing research has largely ignored how these biases behave when they occur simultaneously.

This project aims to fill that gap.

---

## ❗ Problem Statement
LLM-based evaluators exhibit known biases:
- **Position Bias** → Preference for the first response
- **Verbosity Bias** → Preference for longer responses
- **Self-enhancement Bias** → Preference for outputs similar to the model’s style

🔴 **Key Gap:**  
All existing research studies these biases *independently*.  
However, in real-world scenarios, these biases occur **together**.

> Do they:
- Add up?
- Cancel out?
- Amplify each other?

---

## 🎯 Research Objective
To determine whether bias interactions are:
- **Additive** (simple sum)
- **Sub-additive** (partial cancellation)
- **Super-additive** (amplification)

---

## 📚 Related Work
Key prior studies include:
- **Zheng et al. (2023)** – Introduced LLM-as-a-Judge (MT-Bench)
- **Raina et al. (2024)** – Adversarial bias manipulation
- **JudgeBench (2025)** – Benchmarking judge robustness
- **RobustJudge (2025)** – Comprehensive evaluation framework

📌 **Common Limitation:**  
All studies evaluate biases in isolation.

---

## 🗂️ Datasets
We utilize multiple datasets for robust evaluation:

| Dataset | Purpose |
|--------|--------|
| **MT-Bench** | Primary evaluation benchmark (80 questions) |
| **SummEval** | Multi-dimensional evaluation (coherence, fluency, etc.) |
| **FairEval** | Baseline bias comparison |
| **Urdu Subset** | Cross-lingual analysis |

---

## ⚙️ Methodology

### 🔬 Experimental Design
A **2×2 factorial design**:

| Condition | Description |
|----------|------------|
| Short + First | Short response in first position |
| Long + First | Long response in first position |
| Short + Second | Short response in second position |
| Long + Second | Long response in second position |

### 🧪 Pipeline
1. Generate controlled short/long variants of responses  
2. Place responses in different positions  
3. Evaluate using multiple LLM judges:
   - GPT-4o  
   - Claude  
   - LLaMA-3  
   - Mistral  
   - Gemma  
4. Analyze results using:
   - **ANOVA (Analysis of Variance)**
   - Interaction coefficients
   - Flip rate & score drift metrics

---

## 📊 Evaluation Metrics
- **Flip Rate** → How often decisions change when order changes  
- **Score Drift** → Change in scoring due to bias  
- **ANOVA Interaction Term** → Measures bias interaction  
- **Cohen’s Kappa** → Agreement consistency  

---

## 🌍 Cross-Lingual Analysis
A dedicated Urdu dataset is used to evaluate:
- Whether bias interactions are stronger in low-resource languages
- Generalization across linguistic contexts

---

## 📈 Expected Outcomes
We hypothesize that:

> **Biases are super-additive (they amplify each other)**

Expected deliverables:
- Bias interaction matrix across models  
- Statistical classification of interactions  
- Cross-dimensional analysis  
- Cross-lingual insights  
- Practical recommendations  

---

## 🚀 Impact
If biases amplify:
- Current mitigation strategies (which target single biases) are insufficient  
- The field must move toward **compound-aware debiasing**

This work provides:
- The **first empirical study** on bias interaction in LLM judges  
- A **foundation for future research** in reliable AI evaluation  

---

## 👥 Team Members
- Haroon Maqbool  
- Muhammad Azmeer  
- Saria Mahmood  

---

## 🎓 Academic Context
- Course: *Introduction to Natural Language Processing*  
- Instructor: Dr. Faizad Ullah  
- Institution: Forman Christian College (A Chartered University), Lahore  

---

## 📌 Repository Structure (Suggested)
```
├── data/
├── experiments/
├── results/
├── scripts/
├── notebooks/
└── README.md
```

---

## 📎 References
See project proposal and slides for full details.

---

## ⭐ Final Note
This project addresses a fundamental blind spot in LLM evaluation research.  
Understanding compound bias behavior is essential for building **trustworthy AI systems**.

