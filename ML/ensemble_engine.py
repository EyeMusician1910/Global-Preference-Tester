"""
Ensemble inference (SGD + ComplementNB + LogisticRegression).
Feature pipeline must stay aligned with the training notebooks / train.py.

Checkpoints layout:
  ML/checkpoints/artifacts/  → tfidf_full.pkl, tfidf_diff.pkl, scaler.pkl
  ML/checkpoints/model/      → final_*.pkl, ensemble_weights.npy, model_metadata.json
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp

try:
    import textstat

    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

try:
    from textblob import TextBlob

    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _vader = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

try:
    import spacy

    if os.getenv("ENABLE_SPACY_NER") == "1":
        _nlp = spacy.load("en_core_web_sm")
        HAS_SPACY = True
    else:
        _nlp = None
        HAS_SPACY = False
except Exception:
    _nlp = None
    HAS_SPACY = False


STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "what",
    "which",
    "who",
    "how",
    "when",
    "where",
    "why",
    "not",
    "no",
    "so",
    "as",
    "up",
    "out",
    "if",
    "about",
    "into",
    "than",
    "then",
    "some",
    "more",
    "also",
    "just",
    "like",
    "can",
    "all",
    "there",
    "been",
    "being",
    "very",
    "get",
}

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "best",
    "perfect",
    "wonderful",
    "amazing",
    "fantastic",
    "helpful",
    "clear",
    "correct",
    "accurate",
    "right",
    "easy",
    "simple",
    "effective",
    "well",
    "better",
    "improved",
    "nice",
    "useful",
    "benefit",
    "advantage",
    "success",
    "recommend",
    "love",
    "appreciate",
    "thank",
    "happy",
    "pleased",
    "satisfied",
}
NEGATIVE_WORDS = {
    "bad",
    "wrong",
    "error",
    "incorrect",
    "fail",
    "failure",
    "problem",
    "issue",
    "bug",
    "difficult",
    "hard",
    "confusing",
    "confused",
    "unclear",
    "terrible",
    "awful",
    "poor",
    "worse",
    "broken",
    "missing",
    "lack",
    "sorry",
    "unfortunately",
    "unable",
    "cannot",
    "impossible",
    "never",
    "avoid",
    "danger",
    "risk",
    "harm",
}
HEDGE_WORDS = {
    "perhaps",
    "maybe",
    "might",
    "could",
    "possibly",
    "somewhat",
    "arguably",
    "generally",
    "usually",
    "often",
    "sometimes",
    "occasionally",
    "approximately",
    "roughly",
    "about",
    "around",
    "likely",
    "unlikely",
    "probably",
    "presumably",
    "apparently",
    "seemingly",
    "suggest",
    "seem",
    "appear",
    "tend",
    "consider",
    "believe",
    "think",
    "assume",
}
SYCOPHANCY_PHRASES = [
    "great question",
    "excellent question",
    "good question",
    "wonderful question",
    "of course",
    "certainly",
    "absolutely",
    "definitely",
    "sure thing",
    "i'd be happy to",
    "i'd be glad to",
    "i'm happy to help",
    "that's a great",
    "that's an excellent",
]
COMMON_5000_WORDS = STOPWORDS | {
    "said",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "time",
    "year",
    "people",
    "way",
    "day",
    "man",
    "woman",
    "child",
    "world",
    "life",
    "hand",
    "part",
    "place",
    "case",
    "week",
    "company",
    "system",
    "program",
    "question",
    "work",
    "government",
    "number",
    "night",
    "point",
    "home",
    "water",
    "room",
    "mother",
    "area",
    "money",
    "story",
    "fact",
    "month",
    "lot",
    "right",
    "study",
    "book",
    "eye",
    "job",
    "word",
    "business",
    "issue",
    "side",
    "kind",
    "head",
    "house",
    "service",
    "friend",
    "father",
    "power",
    "hour",
    "game",
    "line",
    "end",
    "among",
    "while",
    "might",
    "next",
    "sound",
    "below",
    "saw",
    "something",
    "thought",
    "both",
    "paper",
    "use",
    "together",
    "got",
    "never",
    "know",
    "put",
    "does",
    "told",
    "nothing",
    "sure",
    "come",
    "few",
}


def clean_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"```[\s\S]{0,3000}?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"\*{1,3}([^*]{0,300})\*{1,3}", r"\1", text)
    text = re.sub(r"#{1,6}\s", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) >= 2]
    return " ".join(tokens)


def count_sentences(t):
    return max(1, len(re.split(r"[.!?]+", t))) if t else 0


def vocab_richness(t):
    tokens = t.split()
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def has_code(t):
    return int("```" in t)


def has_bullets(t):
    return int(bool(re.search(r"\n\s*[-•*]\s|\n\s*\d+\.\s", t)))


def has_headers(t):
    return int(bool(re.search(r"\n#{1,4}\s", t)))


def jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap_coeff(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def bigram_overlap(a, b):
    def bg(t):
        tok = t.split()
        return set(zip(tok[:-1], tok[1:]))

    ba, bb = bg(a), bg(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / (len(ba | bb) + 1e-9)


def fk_grade(text):
    if HAS_TEXTSTAT and text.strip():
        try:
            return textstat.flesch_kincaid_grade(text)
        except Exception:
            pass
    words = text.split()
    sentences = max(1, count_sentences(text))
    syllables = sum(max(1, len(re.findall(r"[aeiou]", w.lower()))) for w in words)
    if not words:
        return 0.0
    return 0.39 * (len(words) / sentences) + 11.8 * (syllables / len(words)) - 15.59


def flesch_ease(text):
    if HAS_TEXTSTAT and text.strip():
        try:
            return textstat.flesch_reading_ease(text)
        except Exception:
            pass
    words = text.split()
    sentences = max(1, count_sentences(text))
    syllables = sum(max(1, len(re.findall(r"[aeiou]", w.lower()))) for w in words)
    if not words:
        return 50.0
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))


def avg_syllables_per_word(text):
    words = [w for w in text.split() if w]
    if not words:
        return 0.0
    if HAS_TEXTSTAT:
        try:
            return textstat.avg_syllables_per_word(text)
        except Exception:
            pass
    return sum(max(1, len(re.findall(r"[aeiou]", w.lower()))) for w in words) / len(words)


def avg_words_per_sentence(text):
    words = text.split()
    sentences = max(1, count_sentences(text))
    return len(words) / sentences if words else 0.0


def count_headers(t):
    return len(re.findall(r"\n#{1,6}\s", t))


def count_bullets(t):
    return len(re.findall(r"\n\s*[-•*]\s|\n\s*\d+\.\s", t))


def count_code_blocks(t):
    return len(re.findall(r"```", t)) // 2


def count_paragraphs(t):
    parts = re.split(r"\n\s*\n", t.strip())
    return max(1, len([p for p in parts if p.strip()]))


def formatting_richness(t):
    return count_headers(t) + count_bullets(t) + count_code_blocks(t)


def sentiment_polarity(text):
    if not text.strip():
        return 0.0
    if HAS_VADER:
        try:
            return _vader.polarity_scores(text)["compound"]
        except Exception:
            pass
    if HAS_TEXTBLOB:
        try:
            return TextBlob(text).sentiment.polarity
        except Exception:
            pass
    words = text.lower().split()
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    return (pos - neg) / len(words)


def positive_word_ratio(clean_text_tokens):
    tok = clean_text_tokens.split()
    return sum(1 for w in tok if w in POSITIVE_WORDS) / len(tok) if tok else 0.0


def negative_word_ratio(clean_text_tokens):
    tok = clean_text_tokens.split()
    return sum(1 for w in tok if w in NEGATIVE_WORDS) / len(tok) if tok else 0.0


def hedge_word_ratio(clean_text_tokens):
    tok = clean_text_tokens.split()
    return sum(1 for w in tok if w in HEDGE_WORDS) / len(tok) if tok else 0.0


def sycophancy_flag(raw_text):
    first_100 = raw_text.lower()[:100]
    return int(any(p in first_100 for p in SYCOPHANCY_PHRASES))


def prompt_keyword_coverage(prompt_clean, response_clean):
    p_words = set(prompt_clean.split())
    r_words = set(response_clean.split())
    if not p_words:
        return 0.0
    return len(p_words & r_words) / len(p_words)


def prompt_question_count(raw_prompt):
    return raw_prompt.count("?")


def prompt_question_type(raw_prompt):
    p = raw_prompt.lower().strip()
    if re.search(r"\b(compare|vs|versus|difference between)\b", p):
        return 6
    if re.search(r"\b(list|enumerate|give me)\b", p):
        return 7
    if re.search(r"\b(write|create|generate|draft|make)\b", p):
        return 4
    if re.search(r"\bexplain\b", p):
        return 5
    if p.startswith("how"):
        return 1
    if p.startswith("why"):
        return 2
    if p.startswith("what"):
        return 3
    return 0


def has_code_in_prompt(raw_prompt):
    return int("```" in raw_prompt or bool(re.search(r"`[^`]+`", raw_prompt)))


def named_entity_overlap(prompt_raw, response_raw):
    if not HAS_SPACY:
        return 0.0
    try:
        p_ents = {e.text.lower() for e in _nlp(prompt_raw[:500]).ents}
        r_ents = {e.text.lower() for e in _nlp(response_raw[:500]).ents}
        if not p_ents:
            return 0.0
        return len(p_ents & r_ents) / len(p_ents)
    except Exception:
        return 0.0


def trigram_overlap(a, b):
    def tg(t):
        tok = t.split()
        return set(zip(tok[:-2], tok[1:-1], tok[2:])) if len(tok) >= 3 else set()

    ta, tb = tg(a), tg(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / (len(ta | tb) + 1e-9)


def number_count(text):
    return len(re.findall(r"\b\d+\.?\d*\b", text))


def rare_word_ratio(clean_text_tokens):
    tok = [w for w in clean_text_tokens.split() if len(w) >= 2]
    if not tok:
        return 0.0
    return sum(1 for w in tok if w not in COMMON_5000_WORDS) / len(tok)


def unique_bigram_ratio(clean_text_tokens):
    tok = clean_text_tokens.split()
    if len(tok) < 2:
        return 0.0
    bigrams = list(zip(tok[:-1], tok[1:]))
    return len(set(bigrams)) / len(bigrams)


def unique_content_ratio(response_clean, other_clean):
    r_words = response_clean.split()
    o_words = set(other_clean.split())
    if not r_words:
        return 0.0
    return sum(1 for w in r_words if w not in o_words) / len(r_words)


def keyword_coverage_of_other(a_clean, b_clean):
    b_words = set(b_clean.split())
    a_words = set(a_clean.split())
    if not b_words:
        return 1.0
    return len(b_words & a_words) / len(b_words)


def len_ratio_squared(len_a, len_b):
    return (len_a / (len_b + 1)) ** 2


def extreme_verbosity_flag(len_a, len_b, threshold=3.0):
    return int(len_a > threshold * (len_b + 1))


def prompt_to_response_ratio(len_prompt, len_response):
    return len_response / (len_prompt + 1)


class EnsemblePreferencePredictor:
    def __init__(self, artifact_dir: Path | str, model_dir: Path | str):
        self.artifact_dir = Path(artifact_dir)
        self.model_dir = Path(model_dir)
        self.tfidf_full = None
        self.tfidf_diff = None
        self.scaler = None
        self.model_sgd = None
        self.model_nb = None
        self.model_lr = None
        self.ens_weights: np.ndarray | None = None
        self.metadata: dict = {}
        self.n_dense = 0

    def load(self) -> None:
        self.tfidf_full = joblib.load(self.artifact_dir / "tfidf_full.pkl")
        self.tfidf_diff = joblib.load(self.artifact_dir / "tfidf_diff.pkl")
        self.scaler = joblib.load(self.artifact_dir / "scaler.pkl")
        self.model_sgd = joblib.load(self.model_dir / "final_sgd.pkl")
        self.model_nb = joblib.load(self.model_dir / "final_nb.pkl")
        self.model_lr = joblib.load(self.model_dir / "final_logreg.pkl")
        self.ens_weights = np.load(self.model_dir / "ensemble_weights.npy")
        with open(self.model_dir / "model_metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.n_dense = int(getattr(self.scaler, "n_features_in_", len(self.scaler.scale_)))
        meta_dense = self.metadata.get("n_dense_features")
        if meta_dense is not None and int(meta_dense) != self.n_dense:
            raise RuntimeError(
                f"Artifact mismatch: scaler expects {self.n_dense} dense features, "
                f"metadata says {meta_dense}."
            )

    def build_feature_vector(
        self, prompt: str, response_a: str, response_b: str
    ) -> tuple[sp.csr_matrix, dict]:
        if self.tfidf_full is None:
            raise RuntimeError("Call load() before predict.")

        p_c = clean_text(prompt)
        ra_c = clean_text(response_a)
        rb_c = clean_text(response_b)

        text_full = f"PROMPT: {p_c} RESPONSE_A: {ra_c} RESPONSE_B: {rb_c}"
        sa, sb = set(ra_c.split()), set(rb_c.split())
        text_diff = " ".join(sorted(sa - sb)) + " " + " ".join(sorted(sb - sa))

        X_full_part = self.tfidf_full.transform([text_full])
        X_diff_part = self.tfidf_diff.transform([text_diff])

        len_a = len(response_a.split())
        len_b = len(response_b.split())
        len_p = len(prompt.split())
        vr_a = vocab_richness(ra_c)
        vr_b = vocab_richness(rb_c)
        sc_a = count_sentences(response_a)
        sc_b = count_sentences(response_b)

        f_orig = [
            len_a,
            len_b,
            len_p,
            len_a / (len_b + 1),
            abs(len_a - len_b),
            int(len_a > len_b),
            vr_a,
            vr_b,
            vr_a - vr_b,
            sc_a,
            sc_b,
            sc_a - sc_b,
            has_code(response_a),
            has_code(response_b),
            has_bullets(response_a),
            has_bullets(response_b),
            has_headers(response_a),
            has_headers(response_b),
        ]
        f_sim = [
            jaccard(ra_c, rb_c),
            overlap_coeff(ra_c, rb_c),
            bigram_overlap(ra_c, rb_c),
            jaccard(p_c, ra_c),
            jaccard(p_c, rb_c),
        ]
        f_read = [
            fk_grade(response_a),
            fk_grade(response_b),
            fk_grade(response_a) - fk_grade(response_b),
            flesch_ease(response_a),
            flesch_ease(response_b),
            flesch_ease(response_a) - flesch_ease(response_b),
            avg_syllables_per_word(response_a),
            avg_syllables_per_word(response_b),
            avg_syllables_per_word(response_a) - avg_syllables_per_word(response_b),
            avg_words_per_sentence(response_a),
            avg_words_per_sentence(response_b),
            avg_words_per_sentence(response_a) - avg_words_per_sentence(response_b),
        ]
        f_struct = [
            count_headers(response_a),
            count_headers(response_b),
            count_headers(response_a) - count_headers(response_b),
            count_bullets(response_a),
            count_bullets(response_b),
            count_bullets(response_a) - count_bullets(response_b),
            count_code_blocks(response_a),
            count_code_blocks(response_b),
            count_code_blocks(response_a) - count_code_blocks(response_b),
            count_paragraphs(response_a),
            count_paragraphs(response_b),
            count_paragraphs(response_a) - count_paragraphs(response_b),
            formatting_richness(response_a),
            formatting_richness(response_b),
            formatting_richness(response_a) - formatting_richness(response_b),
        ]
        f_sent = [
            sentiment_polarity(response_a),
            sentiment_polarity(response_b),
            sentiment_polarity(response_a) - sentiment_polarity(response_b),
            positive_word_ratio(ra_c),
            positive_word_ratio(rb_c),
            positive_word_ratio(ra_c) - positive_word_ratio(rb_c),
            negative_word_ratio(ra_c),
            negative_word_ratio(rb_c),
            negative_word_ratio(ra_c) - negative_word_ratio(rb_c),
            hedge_word_ratio(ra_c),
            hedge_word_ratio(rb_c),
            hedge_word_ratio(ra_c) - hedge_word_ratio(rb_c),
            sycophancy_flag(response_a),
            sycophancy_flag(response_b),
            sycophancy_flag(response_a) - sycophancy_flag(response_b),
        ]
        f_align = [
            prompt_keyword_coverage(p_c, ra_c),
            prompt_keyword_coverage(p_c, rb_c),
            prompt_keyword_coverage(p_c, ra_c) - prompt_keyword_coverage(p_c, rb_c),
            prompt_question_count(prompt),
            prompt_question_type(prompt),
            has_code_in_prompt(prompt),
            trigram_overlap(ra_c, rb_c),
            trigram_overlap(p_c, ra_c),
            trigram_overlap(p_c, rb_c),
            named_entity_overlap(prompt, response_a),
            named_entity_overlap(prompt, response_b),
            named_entity_overlap(prompt, response_a) - named_entity_overlap(prompt, response_b),
        ]
        nca = number_count(response_a)
        ncb = number_count(response_b)
        rwa = rare_word_ratio(ra_c)
        rwb = rare_word_ratio(rb_c)
        f_spec = [
            nca,
            ncb,
            nca - ncb,
            rwa,
            rwb,
            rwa - rwb,
            unique_bigram_ratio(ra_c),
            unique_bigram_ratio(rb_c),
            unique_bigram_ratio(ra_c) - unique_bigram_ratio(rb_c),
            nca / (len(ra_c.split()) + 1) + rwa,
            ncb / (len(rb_c.split()) + 1) + rwb,
            (nca / (len(ra_c.split()) + 1) + rwa) - (ncb / (len(rb_c.split()) + 1) + rwb),
        ]
        f_verb = [
            np.log1p(len_a),
            np.log1p(len_b),
            np.log1p(len_a) - np.log1p(len_b),
            len_ratio_squared(len_a, len_b),
            extreme_verbosity_flag(len_a, len_b),
            extreme_verbosity_flag(len_b, len_a),
            prompt_to_response_ratio(len_p, len_a),
            prompt_to_response_ratio(len_p, len_b),
            prompt_to_response_ratio(len_p, len_a) - prompt_to_response_ratio(len_p, len_b),
        ]
        f_cross = [
            unique_content_ratio(ra_c, rb_c),
            unique_content_ratio(rb_c, ra_c),
            keyword_coverage_of_other(ra_c, rb_c),
            keyword_coverage_of_other(rb_c, ra_c),
            1.0 - jaccard(ra_c, rb_c),
        ]
        p_q_cnt = prompt_question_count(prompt)
        f_pcomp = [
            len_p,
            np.log1p(len_p),
            count_sentences(prompt),
            p_q_cnt,
            len_p * (p_q_cnt + 1),
            int(count_sentences(prompt) > 1),
        ]

        all_dense = f_orig + f_sim + f_read + f_struct + f_sent + f_align + f_spec + f_verb + f_cross + f_pcomp

        dense_vec = np.array([all_dense], dtype=np.float32)
        expected_dense = int(getattr(self.scaler, "n_features_in_", len(self.scaler.scale_)))
        if dense_vec.shape[1] != expected_dense:
            dense_names = self.metadata.get("dense_feature_names", [])
            if dense_vec.shape[1] + 1 == expected_dense and dense_names[-1:] == ["prompt_multi_sent"]:
                legacy_extra = np.array([[int(count_sentences(prompt) > 1)]], dtype=np.float32)
                dense_vec = np.hstack([dense_vec, legacy_extra])
            else:
                raise ValueError(
                    f"Dense feature mismatch: built {dense_vec.shape[1]}, "
                    f"but scaler expects {expected_dense}."
                )

        X_dense_scaled = sp.csr_matrix(self.scaler.transform(dense_vec))
        X_vec = sp.hstack([X_full_part, X_diff_part, X_dense_scaled], format="csr")

        insights = {
            "response_a_word_count": len_a,
            "response_b_word_count": len_b,
            "longer_response": "Model A" if len_a > len_b else "Model B",
            "response_a_has_code": bool(has_code(response_a)),
            "response_b_has_code": bool(has_code(response_b)),
            "response_a_has_bullets": bool(has_bullets(response_a)),
            "response_b_has_bullets": bool(has_bullets(response_b)),
            "response_a_has_headers": bool(has_headers(response_a)),
            "response_b_has_headers": bool(has_headers(response_b)),
            "vocabulary_similarity": round(jaccard(ra_c, rb_c), 3),
            "fk_grade_a": f_read[0],
            "fk_grade_b": f_read[1],
            "sentiment_a": f_sent[0],
            "sentiment_b": f_sent[1],
            "prompt_kw_coverage_a": f_align[0],
            "prompt_kw_coverage_b": f_align[1],
            "number_count_a": f_spec[0],
            "number_count_b": f_spec[1],
        }
        return X_vec, insights

    def predict_proba(self, prompt: str, response_a: str, response_b: str) -> np.ndarray:
        """Return shape (3,) probabilities for classes A wins, B wins, tie."""
        X_vec, _ = self.build_feature_vector(prompt, response_a, response_b)
        assert self.ens_weights is not None and self.model_sgd is not None
        n_tfidf = X_vec.shape[1] - self.n_dense
        X_tfidf_vec = X_vec[:, :n_tfidf]
        w_sgd, w_nb, w_lr = self.ens_weights[0], self.ens_weights[1], self.ens_weights[2]
        p_sgd = self.model_sgd.predict_proba(X_vec)
        p_nb = self.model_nb.predict_proba(X_tfidf_vec)
        p_lr = self.model_lr.predict_proba(X_vec)
        p_ens = w_sgd * p_sgd + w_nb * p_nb + w_lr * p_lr
        return np.asarray(p_ens[0], dtype=np.float64)
