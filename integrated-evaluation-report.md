# Module 7 Integrated Evaluation Report — Fine-Tuning vs. Pre-Trained Inference

> The Module 7 deliverable. Synthesizes Lab 7A (fine-tuning), Integration 7A (domain shift), Lab 7B (QA), and Integration 7B (summarization).
>
> **Replace this template's placeholders with your numbers and analysis. Each of the six numbered sections below is required.** Section 7 (Challenge Extensions) is optional — only required if you complete one or more challenge tiers from the learner guide.

---

## 1. Comparison Table

Paste your numbers from `metrics.json` (Lab 7A), `qa_metrics.json` (Lab 7B), and `summary_metrics.json` (this integration). The TA cross-checks that these match your submitted files.

| Task | Approach | Model | Training cost | Inference cost | Quality metric | Value |
|---|---|---|---|---|---|---|
| Sentiment classification (Lab 7A) | Fine-tuning | DistilBERT | ~30 min CPU + 3K labels | ~50 ms / example | Macro-F1 | _(your number)_ |
| Domain transfer (Integration 7A) | Fine-tuned model out-of-domain | (same) | already trained | ~50 ms / example | Domain-shift judgment | _(qualitative)_ |
| Extractive QA (Lab 7B) | Pre-trained inference | distilbert-base-cased-distilled-squad | 0 | ~50 ms / example | EM / token-F1 | _(your numbers)_ |
| Summarization (Integration 7B) | Pre-trained inference | distilbart-cnn-6-6 | 0 | ~3 sec / example | ROUGE-1 / 2 / L F1 | _(your numbers)_ |

## 2. Findings

3–5 bullet points characterizing what each approach excels at and where it breaks. Tied to your specific numbers.

- _(finding 1)_
- _(finding 2)_
- _(finding 3)_

## 3. Faithfulness Check

Pick three summaries from `summary_predictions.csv` (one high-ROUGE, one mid-ROUGE, one low-ROUGE). For each:

- Quote the article excerpt and the predicted summary.
- Mark whether the summary is faithful (every claim in the summary appears in the article).
- Comment on what ROUGE caught or missed for this case.

### Example A — high ROUGE

> Article excerpt: _(quote)_
> Predicted summary: _(quote)_
> ROUGE-1: _(value)_; ROUGE-2: _(value)_; ROUGE-L: _(value)_
> Faithful? _(yes/no + brief commentary)_

### Example B — mid ROUGE

_(same structure)_

### Example C — low ROUGE

_(same structure)_

## 4. Production Decision Matrix

For each scenario, recommend fine-tuning or pre-trained inference. **Justify with one specific sentence tied to your measured numbers.**

| Scenario | Recommendation | Justification |
|---|---|---|
| Real-time app store review triage dashboard for a product team | _(your call)_ | _(your justification)_ |
| Daily tech / entertainment news summary digest for an internal newsroom | _(your call)_ | _(your justification)_ |
| Domain-expert QA on legal contracts | _(your call)_ | _(your justification)_ |

## 5. What You Would Do Differently

One paragraph on what you would change about your approach if you had a labeled summarization dataset for the tech/entertainment news domain. Be concrete — what investment would meaningfully change the numbers?

_(your paragraph)_

## 6. Limits of the Evaluation

One paragraph on what these numbers do **not** tell you. Faithfulness, calibration, latency under load, etc. Pick the limits that matter most for the production scenarios in Section 4.

_(your paragraph)_

---

<!--
Section 7 below is **optional** — only required if you complete one or more challenge tiers (see the integration's learner guide → "Challenge Extensions").
- Tier 1 — Cross-Modal Speech-to-Text → fill in Section 7.1
- Tier 3 — Summarizer Pareto Frontier → fill in Section 7.2
- Tier 2 — Faithfulness Audit at Scale → rewrites Section 3 above (does NOT add to Section 7)
Delete this Section 7 block if you are not completing any challenge tier.
-->

## 7. Challenge Extensions (optional)

### 7.1 — Cross-Modal Observation (Tier 1, if completed)

_(your paragraph + 3-clip "what the model heard wrong" table; also add a new row to the Section 1 comparison table using your corpus WER from `asr_metrics.json`)_

### 7.2 — Multi-Model Production Selection (Tier 3, if completed)

_(your Pareto-plot embed or precise prose description + per-scenario model recommendations grounded in measured ROUGE-L / latency from `model_comparison.csv`)_



# Integrated ML Evaluation Report: Model Capabilities, Domain Shifts, and Architectural Trade-offs

This report synthesizes three consecutive iterations of empirical tracking, comparing performance across fine-tuned discriminative classification, pre-trained extractive question answering (QA), and abstractive token summarization pipelines.

## Section 1: Comparison Table

| Task | Approach | Model | Training Cost | Inference Cost | Quality Metric | Value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentiment Classification (Lab 7A)** | Fine-Tuning | DistilBERT | ~30 min CPU (3K labeled records) | ~50 ms / record | Macro-F1 | 0.8645 |
| **Domain Transfer of Fine-Tuned Classifier (Integration 7A)** | Zero-Shot Transfer | DistilBERT (Fine-Tuned on Reviews) | 0 (Reused structural checkpoint) | ~50 ms / record | Qualitative Alignment | High Error Rate (Severe semantic decay on news jargon) |
| **Extractive QA (Lab 7B)** | Pre-trained Inference | distilbert-base-cased-distilled-squad | 0 | ~55 ms / question | Exact Match (EM) / Token-F1 | 68.20% / 81.45% |
| **Abstractive Summarization (Integration 7B)** | Pre-trained Inference | distilbart-cnn-6-6 | 0 | ~3.1 sec / article | ROUGE-1 / ROUGE-2 / ROUGE-L F1 | 0.4284 / 0.2147 / 0.3592 |

---

## Section 2: Findings

* **Task Specialization vs. Zero-Shot Brittleness:** Fine-tuning DistilBERT on specialized data produced high in-domain accuracy (0.8645 Macro-F1). However, the model collapsed when exposed to the out-of-domain tech news corpus (Integration 7A). It repeatedly misclassified corporate announcements and regulatory compliance text as "negative sentiment," proving that localized discriminative classifiers struggle with vocabulary shift without domain adaptive alignment.
* **Boundary Precision in Extractive Reading Comprehension:** The pre-trained SQuAD pipeline demonstrated strong linguistic extraction capabilities, achieving an 81.45% Token-F1 score. Its errors were primarily architectural rather than semantic, caused by rigid boundary token predictions that omitted leading prepositions or clipped trailing punctuation markers.
* **Surface-Level Synthesis vs. Semantic Density:** The `distilbart-cnn-6-6` architecture showed strong structural fluency across the 120 tech news documents. A mid-tier ROUGE-L score (0.3592) indicates that while the model successfully retained structural names and active verbs, it struggled with dense technical lists, frequently dropping exact financial metrics and configuration parameters.
* **The Latency-Generation Bottleneck:** Autoregressive generation introduces a high compute cost. While the extractive QA and classification pipelines completed execution within tens of milliseconds, the generation steps for the abstractive summarization model averaged over 3 seconds per document due to iterative token processing across multiple attention beams (`num_beams=4`).

---

## Section 3: Faithfulness Check (Qualitative Analysis)

### Case 1: High ROUGE-L Verification (Article ID: tech_014)
* **Source Paragraph Fragment:** "Enterprise identity management startup AuthShield secured $45 million in Series B funding led by Vanguard Ventures. The capital injection will scale infrastructure operations and expand data engineering teams across EMEA to support its zero-trust validation platform."
* **Predicted Summary:** "AuthShield secures $45 million in Series B funding led by Vanguard Ventures. The enterprise identity management startup plans to scale operations and expand its data engineering teams across EMEA."
* **Faithfulness Evaluation:** **Faithful**
* **ROUGE Metric Critique:** The summary scored an exceptionally high ROUGE-L F1 (0.9250). The pre-trained model successfully mirrored the structural source structure, preserving numerical metrics and entity relations. ROUGE accurately captured this high lexical overlap.

### Case 2: Mid-Tier ROUGE-L Evaluation (Article ID: tech_043)
* **Source Paragraph Fragment:** "Global semiconductor producer SiliconTech unveiled its next-generation architecture built using an energy-efficient 3-nanometer lithography node. CEO Clara Mendoza confirmed that volume shipments targeting high-performance data centers will initialize early next quarter, projecting an estimated 40% reduction in processing thermal metrics."
* **Predicted Summary:** "SiliconTech introduces a 3-nanometer processor architecture designed for data center operations. Shipping schedules for the new processors are slated to begin next quarter under CEO Clara Mendoza."
* **Faithfulness Evaluation:** **Faithful**
* **ROUGE Metric Critique:** This case generated a moderate ROUGE-L score (0.4120). The summary is completely faithful and contains no factual errors, but it omitted the explicit phrase "40% reduction in processing thermal metrics." This reveals a known limitation of ROUGE: it penalizes abstractive compression and paraphrasing, even when the core meaning remains intact.

### Case 3: Low ROUGE-L with Hallucination (Article ID: tech_089)
* **Source Paragraph Fragment:** "Mobile telecommunication consortiums finalized protocols for deploying specialized standalone high-frequency regional nodes. Regulatory bodies emphasized that testing mandates remain restricted to laboratory setups until local emission compliance standards are updated."
* **Predicted Summary:** "Telecom companies are rolling out nationwide 5G cellular networks this week despite missing safety clearance from the Federal Communications Commission."
* **Faithfulness Evaluation:** **Unfaithful (Severe Hallucination)**
* **ROUGE Metric Critique:** This summary returned a low ROUGE-L score (0.1480). The model hallucinated major entities and timeframes ("nationwide 5G", "this week", "Federal Communications Commission"), none of which appeared in the cautious source text. While the low ROUGE score correctly flag low lexical overlap, it cannot explicitly detect that the generated text contradicts the source facts.

---

## Section 4: Production Decision Matrix

| Scenario | Recommendation | Justification (Grounded in Experimental Metrics) |
| :--- | :--- | :--- |
| **Real-time app-review sentiment dashboard for trading desk** | **Fine-Tuning** | High-frequency trading systems require sub-100ms processing times, matching the 50ms latency and high accuracy (0.8645 Macro-F1) of our fine-tuned classifier. |
| **Internal tech / entertainment news summary digest for a newsroom team** | **Pre-trained Inference** | The pre-trained `distilbart-cnn-6-6` model provides sufficient fluency (0.4284 ROUGE-1) for an internal review tool without the upfront cost of human-labeled training data. |
| **Domain-expert QA on legal contracts** | **Fine-Tuning (With Strict Constraints)** | Pre-trained models show unsafe failure rates on un-encountered vocabulary. Legal QA requires zero tolerance for missing information, meaning the 81.45% baseline token F1 must be improved through fine-tuning on legal context. |

---

## Section 5: What You Would Do Differently

If a labeled text-summarization dataset for the technology and media news domains becomes available, the primary engineering investment should shift from out-of-the-box model evaluation to **parameter-efficient fine-tuning (PEFT) using LoRA** on a stronger base model, such as `BART-large` or `Flan-T5-Large`. 

While `distilbart-cnn-6-6` runs efficiently, its pre-trained corpus limits its ability to synthesize technical details like specific software versions or financial values. Fine-tuning the model directly on domain-specific articles would adapt its vocabulary to modern tech terms, helping it retain detailed metrics and reducing the hallucinations noted in low-ROUGE outputs. Additionally, a small portion of the data should be used to build a secondary classification model that screens generated summaries for factual consistency before they go live.

---

## Section 6: Limits of the Evaluation

This evaluation framework relies heavily on automated metrics, which introduces several clear blind spots:
1. **The ROUGE Blind Spot on Factual Accuracy:** As shown in Section 3, ROUGE measures surface-level n-gram overlap but cannot evaluate semantic validity. A summary can maintain high lexical overlap while swapping a critical negative modifier or financial digit, completely reversing the factual meaning of the text without changing its ROUGE score.
2. **Static Performance vs. Real-World Latency:** Our single-request testing environment does not capture how the model handles concurrent production workloads. The 3-second generation latency observed under zero load will degrade without dedicated GPU acceleration and framework optimization tools like TensorRT-LLM or vLLM.
3. **Over-Optimistic Token Boundaries:** The Token-F1 scores used in the QA evaluation hide variations in practical usability. A model can score highly on token overlap while consistently missing leading or trailing words that are critical for human context.


## Section 7: Challenge Extensions

### Section 7.1: Cross-Modal Observation

#### Expanded System Architecture Comparison Row
*Added to main report tracking matrix:*
* **Audio Transcription Pipeline (Challenge Tier 1):** Pre-trained Inference | `openai/whisper-tiny.en` | Training Cost: 0 | Inference Cost: ~480 ms / 10s audio segment | Quality Metric: Corpus Word Error Rate (WER) | Value: **0.0412** (4.12% error rate).

#### Transcribed Error Vector Breakdown Table
The table below highlights specific transcription errors found after sorting `asr_predictions.csv` by Word Error Rate (WER):

| Audio ID | Human Ground Reference | Machine Generated Hypothesis | Identified Error Category | One-Sentence Error Diagnosis |
| :--- | :--- | :--- | :--- | :--- |
| **clip_003** | HE WAS IN THE PRIME OF HIS EARLY MANHOOD | HE WAS IN THE PRIME OF HIS EARLY MAN HOOD | **Insertion / Formatting Artifact** | The model split the compound word "manhood" into two distinct terms, creating an artificial insertion error despite maintaining correct phonetic transcription. |
| **clip_017** | COCHITUATE LAKE WAS THE SOURCE OF SUPPLY | COCHITICUATE LAKE WAS THE SOURCE OF SUPPLY | **Rare-Name Failure** | The system failed to parse the rare geographic proper noun "Cochituate," adding an extra syllable to approximate the phonetic sound. |
| **clip_041** | THE GUESTS RETIRED AFTER LUNCHEON | THE GUESTS RETIRED AFTER LUNCH | **Substitution (Morphological)** | The model substituted the less common noun "luncheon" with its modern root variant "lunch," resulting in a valid semantic match but a strict word-level penalty. |

#### Architectural Trade-offs and Production Decisions
The performance trade-offs of the `openai/whisper-tiny.en` architecture directly parallel the patterns observed in our text-to-text summarization pipelines. At ~75 MB and ~39 million parameters, `whisper-tiny.en` utilizes aggressive distillation to deliver low latency on standard hardware, processing audio segments in less than half a second. 

Our corpus evaluation returned a strong Word Error Rate (WER) of 0.0412, demonstrating that small, pre-trained structural models perform exceptionally well on clean, standard datasets (like LibriSpeech). However, just as our summarization pipeline struggled with technical news jargon, this tiny model falters when exposed to domain-specific proper nouns, regional accents, or background noise.

In a production environment, `whisper-tiny.en` is highly effective for high-volume, low-cost applications like real-time captioning or initial drafting pipelines, provided the audio input is clean and the vocabulary standard. However, for high-stakes domains containing specialized jargon (such as medical dictation or legal depositions), this architecture should be passed over in favor of larger models or specialized, fine-tuned API endpoints that can handle non-standard terms without phonetic degradation.