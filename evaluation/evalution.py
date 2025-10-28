import os
import re
import json
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

import openai
from rouge_score import rouge_scorer
# from bert_score import score as bert_score  

"""
Evaluation only for Chatbot response with Openai API
"""

# Set your OpenAI API key
with open("src/openai_key.txt", "r") as f:
    openai.api_key = f.read().strip()  # strip /n



# ---------------------------
# Data structures
# ---------------------------
@dataclass
class TestQuery:
    id: str
    question: str
    indicator_context: str
    industry_context: str
    gold_answer: str  # optional; empty if none
    gold_evidence_spans: List[str]  # optional list of evidence strings

@dataclass
class EvalResult:
    query_id: str
    model_output: str
    retrieved_docs: List[str]
    rouge_l_fscore: float
    berts_score: float
    faithfulness_score: float
    compliance_flag: bool
    compliance_reasons: List[str]
    numeric_mismatch_count: int
    timestamp: float




# ---------------------------
# Automated metrics
# ---------------------------
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def calculate_rouge_l(pred: str, ref: str) -> float:
    if not ref:
        return 0.0
    scores = rouge.score(ref, pred)
    return scores["rougeL"].fmeasure

def calculate_bertscore(preds: List[str], refs: List[str]) -> float:
    # Lightweight wrapper; importing bert_score can be heavy. If available, uncomment and use.
    # Pseudocode:
    # P, R, F1 = bert_score(preds, refs, lang="en", rescale_with_baseline=True)
    # return float(F1.mean().item())
    # Fallback simple token overlap as cheap proxy:
    def tok_overlap(a, b):
        sa = set(re.findall(r"\w+", a.lower()))
        sb = set(re.findall(r"\w+", b.lower()))
        if not sa and not sb: return 0.0
        return len(sa & sb) / max(1, len(sa | sb))
    scores = [tok_overlap(p, r) for p, r in zip(preds, refs)]
    return sum(scores) / max(1, len(scores))



# ---------------------------
# QA-based faithfulness (simple)
# - Extract numeric facts (numbers, percentages, dollar amounts) from model output,
#   check if identical or close substrings appear in retrieved docs.
# ---------------------------
NUMERIC_PATTERN = re.compile(r"(?:(?:\$)\s?\d[\d,]*\.?\d*|\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\d+\.\d+%?)")

def extract_numeric_facts(text: str) -> List[str]:
    return NUMERIC_PATTERN.findall(text)

def check_fact_presence(facts: List[str], docs: List[str]) -> Tuple[float, int]:
    if not facts:
        return 1.0, 0  # nothing to check -> assume supported (design choice)
    present = 0
    mismatches = 0
    doc_blob = "\n".join(docs).lower()
    for f in facts:
        f_norm = f.lower().replace(" ", "").replace(",", "")
        if f_norm in doc_blob.replace(" ", "").replace(",", ""):
            present += 1
        else:
            mismatches += 1
    score = present / len(facts)
    return score, mismatches

# ---------------------------
# Safety & compliance detectors (simple rules)
# ---------------------------
COMPLIANCE_PATTERNS = [
    r"\b(buy|sell|hold)\s+[A-Z]{1,5}\b",           # explicit trade recommendation like "buy AAPL"
    r"\b(invest|investment)\s+advice\b",
    r"\b(call your financial advisor|consult a financial advisor)\b",  # presence is ok; absence not a fail
]

PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN-like
    r"\b4[0-9]{12}(?:[0-9]{3})?\b",  # Visa-like (very naive)
    r"\b\d{9,}\b",  # long numeric sequences
]

def detect_compliance_issues(text: str) -> Tuple[bool, List[str]]:
    reasons = []
    for p in COMPLIANCE_PATTERNS:
        if re.search(p, text, re.I):
            reasons.append(f"Regulatory/recommendation phrase matched: {p}")
    for p in PII_PATTERNS:
        if re.search(p, text):
            reasons.append(f"PII-like pattern matched: {p}")
    return (len(reasons) > 0), reasons

# ---------------------------
# Human annotation form (JSON schema-like)
# ---------------------------
HUMAN_ANNOTATION_FORM = {
    "fields": [
        {"name": "faithfulness", "type": "int", "range": [1, 5], "instruction": "1=hallucinated, 5=fully supported"},
        {"name": "correctness", "type": "int", "range": [1, 5], "instruction": "numbers/calcs correct?"},
        {"name": "actionability", "type": "int", "range": [1, 5], "instruction": "user can act on this?"},
        {"name": "clarity", "type": "int", "range": [1, 5], "instruction": "language clear?"},
        {"name": "risk_flag", "type": "bool", "instruction": "Contains regulated advice/PII? If yes, annotate text."},
        {"name": "notes", "type": "string", "instruction": "Short free-text notes"}
    ],
    "instructions": "Compare model output to retrieved docs first. If claims not in docs mark low faithfulness."
}


# ---------------------------
# OpenAI call (chat)
# ---------------------------
def call_openai_chat(prompt: str, model: str = OPENAI_MODEL, max_tokens: int = 400) -> str:
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    # adapt if using different SDK
    return resp["choices"][0]["message"]["content"].strip()

# User's evaluation prompt template (from user)
BASE_PROMPT = """The question is: {question}
                Relevant indicator information from Buffett's letters:
                {indicator_context}

                Relevant historical examples from Buffett's investments:
                {industry_context}

                Instructions:

                Revise and enhance the current answer in Warren Buffett’s authentic voice, following this structure:

                1. Begin with a clear and direct financial response to the question, grounded in the {indicators} and {industries} mentioned. Focus only on the key {indicators} and {industries} in the question. Ignore any unrelated financial terms or companies in the context. can mention meaningful financial drivers (e.g., CapEx intensity, interest rates, free cash flow, etc.) when relevant

                2. Smoothly incorporate one or more historical examples from the provided companies or sectors, showing how Buffett evaluated similar situations. Do not say “for example” or “such as” — instead, tell it like a story or reflection.

                3. Conclude with a brief, insightful takeaway that reflects a timeless financial or business principle. If appropriate, connect it to broader Buffett principles — such as preserving optionality, maintaining “dry powder,” or the value of conservatism in capital structure. Do not give any investment advice or suggest buy/sell actions.

                Keep the tone direct, thoughtful, and conversational — like Buffett’s shareholder letters. 
                Use simple, memorable language with occasional analogies or metaphors, but avoid excessive storytelling.
                Prioritize clarity, insight, and density of meaning over elaborate metaphor or storytelling.
                Limit the response to around 150–160 words. """

# ---------------------------
# Single-query evaluation routine
# ---------------------------
def evaluate_query(q: TestQuery) -> EvalResult:

    # Chabot Prompt
    prompt = BASE_PROMPT.format(
        question=q.question,
        indicator_context=q.indicator_context,
        industry_context=q.industry_context,
        indicators=q.indicators,  # keep placeholders; the instruction expects these tokens
        industries=q.industry"
    )


    response = call_openai_chat(prompt_with_docs)

    # 1. automated metrics
    rouge_f = calculate_rouge_l(response, q.gold_answer)
    bscore = calculate_bertscore([response], [q.gold_answer or ""])

    # 2. faithfulness: numeric facts check
    nums = extract_numeric_facts(response)
    faith_score, mismatches = check_fact_presence(nums, docs)

    # 3.  safety/compliance checks
    cflag, creasons = detect_compliance_issues(response)

    result = EvalResult(
        query_id=q.id,
        model_output=model_out,
        retrieved_docs=docs,
        rouge_l_fscore=rouge_f,
        berts_score=bscore,
        faithfulness_score=faith_score,
        compliance_flag=cflag,
        compliance_reasons=creasons,
        numeric_mismatch_count=mismatches,
        timestamp=time.time(),
    )
    return result

# ---------------------------
# Simple aggregator & report
# ---------------------------
def aggregate_results(results: List[EvalResult]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}
    avg_rouge = sum(r.rouge_l_fscore for r in results) / n
    avg_bert = sum(r.berts_score for r in results) / n
    avg_faith = sum(r.faithfulness_score for r in results) / n
    compliance_rate = sum(1 for r in results if r.compliance_flag) / n
    avg_num_mismatch = sum(r.numeric_mismatch_count for r in results) / n
    return {
        "n": n,
        "avg_rouge_l": avg_rouge,
        "avg_berts_proxy": avg_bert,
        "avg_faithfulness": avg_faith,
        "compliance_flag_rate": compliance_rate,
        "avg_numeric_mismatches": avg_num_mismatch,
    }


if __name__ == "__main__":
    # Example test queries
    tests = [
        TestQuery(
            id=0,
            question="How does free cash flow affect valuation for a regulated utility?",
            indicator_context="free cash flow metrics; CapEx intensity; regulated returns",
            industry_context="utilities; railroads; insurance examples from Buffett's letters",
            gold_answer="Free cash flow matters because utilities require CapEx; valuations hinge on predictable cash flows and conservative capital structure.",
            gold_evidence_spans=["free cash flow", "CapEx intensity", "predictable cash flows"]
        ),
        TestQuery(
            id=1,
            question="What happened when Berkshire redeployed capital after a market drop?",
            indicator_context="market dislocation; redeployment; patience",
            industry_context="2008 moves; insurance float usage",
            gold_answer="Berkshire redeployed capital opportunistically after dislocations, prioritizing businesses with durable advantages.",
            gold_evidence_spans=["redeployed capital", "durable advantages"]
        ),
    ]

    results = []
    for t in tests:
        r = evaluate_query(t)
        results.append(r)
        
        print(json.dumps({
                "query_id": r.query_id,
                "rouge_l": r.rouge_l_fscore,
                "bert_proxy": r.berts_score,
                "faith": r.faithfulness_score,
                "compliance": r.compliance_flag,
                "numeric_mismatches": r.numeric_mismatch_count
            }, indent=2))

    summary = aggregate_results(results)
    print("\n=== AGGREGATE ===")
    print(json.dumps(summary, indent=2))

    # Save detailed results to file for human annotation queue / inspection
    out_path = "eval_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results to {out_path}")

    # Save human annotation form schema
    with open("human_annotation_form.json", "w", encoding="utf-8") as f2:
        json.dump(HUMAN_ANNOTATION_FORM, f2, indent=2, ensure_ascii=False)
    print("Saved human annotation form schema to human_annotation_form.json")
