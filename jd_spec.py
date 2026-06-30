"""
Structured representation of the Redrob "Senior AI Engineer — Founding Team" JD.

This is NOT a generic keyword list. Every field here was extracted by actually
reading the JD (job_description.docx) and encodes the *intent* behind each
requirement, including the explicit "what we don't want" and "how to read
between the lines" sections that most naive systems ignore.

Keeping this as a separate, human-readable config (rather than burying magic
numbers inside the scorer) is deliberate: at the Stage 5 interview we need to
be able to point to *exactly* which JD sentence justifies each weight.
"""

# ---------------------------------------------------------------------------
# Must-have technical skill families (each family = a set of synonymous /
# adjacent terms; a candidate matches the family if ANY term is present with
# sufficient trust -- see scoring.skill_trust). Matching the family matters
# more than matching the exact string, because the JD explicitly says
# "we don't care which model/vector-db, we care about the operational
# experience."
# ---------------------------------------------------------------------------
MUST_HAVE_SKILL_FAMILIES = {
    "embeddings_retrieval": [
        "sentence-transformers", "sentence transformers", "openai embeddings",
        "bge", "e5", "embeddings", "embedding", "dense retrieval",
        "semantic search", "retrieval", "RAG", "hybrid retrieval",
    ],
    "vector_db": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
        "elasticsearch", "faiss", "vector database", "vector db",
        "hybrid search",
    ],
    "python": ["python"],
    "eval_frameworks": [
        "ndcg", "mrr", "map", "a/b test", "ab test", "offline evaluation",
        "online evaluation", "evaluation framework", "learning to rank",
        "learning-to-rank", "ranking evaluation",
    ],
}

# Nice-to-have, not a rejection criterion either way.
NICE_TO_HAVE_SKILL_FAMILIES = {
    "llm_finetuning": ["lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning"],
    "ltr_models": ["xgboost", "learning-to-rank", "neural ranking", "ranknet", "lambdamart"],
    "hr_tech": ["hr-tech", "hr tech", "recruiting tech", "marketplace"],
    "distributed_systems": ["distributed systems", "large-scale inference", "inference optimization"],
    "open_source": ["open source", "open-source", "github", "publication", "paper"],
}

# Title-family regexes (lower-cased substring match) that indicate the
# candidate is plausibly in the right *kind* of role at all. This is a soft
# signal, not a hard filter -- the JD explicitly rewards non-obvious titles
# ("Senior Engineer | Information Retrieval at scale") that don't say "AI".
CORE_AI_TITLE_MARKERS = [
    "ai engineer", "ml engineer", "machine learning engineer",
    "applied scientist", "research engineer", "data scientist",
    "nlp engineer", "applied ml", "ai specialist", "ai research",
]
ADJACENT_TITLE_MARKERS = [
    "search", "ranking", "retrieval", "recommendation", "backend engineer",
    "data engineer", "software engineer", "full stack",
]

# ---------------------------------------------------------------------------
# Explicit disqualifiers / down-weights, straight from "Things we explicitly
# do NOT want" and the disqualifier paragraph. Severity: HARD = applied as a
# strong multiplicative penalty (never a literal zero-out, since the JD says
# "probably not" rather than "automatically rejected" for most of these, and
# a hard zero would make the system as brittle as a keyword filter).
# ---------------------------------------------------------------------------
CONSULTING_FIRMS = [
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tata consultancy", "hcl", "tech mahindra", "mindtree", "ltimindtree",
]

PURE_RESEARCH_MARKERS = ["research scientist", "research fellow", "phd researcher", "postdoc"]

CV_SPEECH_ROBOTICS_MARKERS = [
    "computer vision engineer", "speech engineer", "robotics engineer",
    "robotics scientist", "cv engineer",
]

TIER1_INDIAN_CITIES = {"pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram", "bengaluru", "bangalore"}
PREFERRED_CITIES = {"pune", "noida"}

# Ideal-candidate envelope distilled from "How to read between the lines".
IDEAL_YOE_LOW, IDEAL_YOE_HIGH = 6, 8          # "6-8 years total experience"
ACCEPTABLE_YOE_LOW, ACCEPTABLE_YOE_HIGH = 5, 9  # the literal "5-9" band in the headline
APPLIED_ML_FRACTION_TARGET = 0.5               # "of which 4-5 are in applied ML/AI roles" (~5/7)

MAX_NOTICE_PERIOD_SOFT = 30   # "we'd love sub-30-day notice"; beyond this, bar gets higher
