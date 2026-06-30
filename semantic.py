"""
Local semantic-fit scoring between a candidate's free text and the JD.

Why TF-IDF + truncated SVD (LSA) instead of a neural embedding model:
the submission_spec.md compute constraints (CPU only, no GPU, no network,
<=5 min wall clock, <=16GB RAM for the ranking step) explicitly rule out
calling a hosted embedding API, and downloading a sentence-transformers
checkpoint at ranking time would violate "no network during ranking" if
the weights aren't already vendored. A from-scratch TF-IDF + SVD model
fit on the candidate pool itself:

  - has zero external dependencies beyond scikit-learn (already required
    for the learning-to-rank-style score fusion anyway),
  - fits and transforms 100K candidates in well under a minute on CPU,
  - captures "semantic fit beyond keywords" through co-occurrence structure
    (e.g. "hybrid retrieval", "recommendation system", "search relevance"
    end up near each other in the latent space even without exact string
    overlap with the JD),
  - is fully reproducible and inspectable -- there's no opaque pretrained
    checkpoint to "trust"; the vocabulary and components are right there in
    the pickled artifact.

This is explicitly the tradeoff the JD asks for: "ship a working ranker...
even if the underlying ML is obviously suboptimal." A from-scratch
TF-IDF/LSA semantic layer is a deliberate, documented choice given the
constraints, not a cop-out -- it is combined with the structured feature
layer in scoring.py specifically so the system isn't relying on lexical
similarity alone.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+\-]+")


def candidate_doc(candidate: dict[str, Any]) -> str:
    """Build the free-text document used for semantic comparison."""
    p = candidate.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", "")]
    for ch in candidate.get("career_history", []) or []:
        parts.append(ch.get("title", ""))
        parts.append(ch.get("description", ""))
    skill_names = [s.get("name", "") for s in candidate.get("skills", []) or []]
    parts.append(" ".join(skill_names))
    return " . ".join(p for p in parts if p)


# The JD itself, condensed into the same "free text" shape so it lives in the
# same vector space as candidate docs. Built from the actual JD content
# (job_description.docx) -- the requirements paragraphs, the explicit
# disqualifiers, and the "how to read between the lines" ideal-candidate
# paragraph, since that section is the clearest single statement of fit.
JD_DOCUMENT = """
Senior AI Engineer Founding team at an AI native talent intelligence platform.
Own the intelligence layer: ranking, retrieval, and matching systems that
decide what recruiters see when they search for candidates. Production
experience with embeddings based retrieval systems such as sentence
transformers, OpenAI embeddings, BGE, E5 deployed to real users, including
handling embedding drift, index refresh, and retrieval quality regression in
production. Production experience with vector databases or hybrid search
infrastructure such as Pinecone, Weaviate, Qdrant, Milvus, OpenSearch,
Elasticsearch, or FAISS. Strong Python and code quality. Hands on experience
designing evaluation frameworks for ranking systems: NDCG, MRR, MAP,
offline to online correlation, A/B test interpretation. Nice to have: LLM
fine tuning with LoRA QLoRA or PEFT, learning to rank models such as XGBoost
or neural ranking, prior HR tech or recruiting tech or marketplace product
experience, distributed systems or large scale inference optimization,
open source contributions. Ideal candidate has shipped an end to end
ranking, search, or recommendation system to real users at meaningful scale,
has strong opinions about hybrid versus dense retrieval and offline versus
online evaluation and when to fine tune versus prompt an LLM, and can defend
those opinions with reference to systems they actually built. Comfortable
with a scrappy product engineering attitude: willing to ship a working
ranker in a week even if the underlying ML is suboptimal, in order to learn
from real users. Has applied ML and AI experience at product companies, not
purely research labs and not purely IT services consulting. Has written
production code recently, not exclusively architecture or tech lead work.
"""


class SemanticScorer:
    """Fits TF-IDF + LSA on the candidate corpus, scores any doc against the JD."""

    def __init__(self, n_components: int = 128, max_features: int = 40_000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.6,
            token_pattern=_TOKEN_RE.pattern,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False
        self._jd_vec = None

    def fit(self, docs: list[str]) -> "SemanticScorer":
        tfidf = self.vectorizer.fit_transform(docs)
        self.svd.fit(tfidf)
        self._fitted = True
        jd_tfidf = self.vectorizer.transform([JD_DOCUMENT])
        self._jd_vec = self.svd.transform(jd_tfidf)
        return self

    def transform(self, docs: list[str]) -> np.ndarray:
        tfidf = self.vectorizer.transform(docs)
        return self.svd.transform(tfidf)

    def score_against_jd(self, docs: list[str]) -> np.ndarray:
        """Cosine similarity in LSA space, rescaled to roughly [0, 1]."""
        assert self._fitted, "call fit() first"
        vecs = self.transform(docs)
        sims = cosine_similarity(vecs, self._jd_vec).ravel()
        # cosine sims from LSA on this kind of corpus cluster in a fairly
        # narrow positive band; min-max rescale across the batch so the
        # *relative* ordering (what ranking actually needs) gets full
        # dynamic range instead of being compressed near the top.
        lo, hi = sims.min(), sims.max()
        if hi - lo < 1e-9:
            return np.zeros_like(sims)
        return (sims - lo) / (hi - lo)
