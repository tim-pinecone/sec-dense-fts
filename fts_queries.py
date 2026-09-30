import json
import re

TEXT_FIELD = "text"

OCCURS = ["SHOULD", "MUST", "MUST NOT"]
OCCUR_PREFIX = {"SHOULD": "", "MUST": "+", "MUST NOT": "-"}
KINDS = ["term(s)", "phrase", "phrase prefix"]
MATCH_OPS = ["$match_phrase", "$match_all", "$match_any"]

LUCENE_KEYWORDS = {"and", "or", "not", "text"}
STOP_WORDS = {
    "a", "an", "the", "of", "on", "in", "to", "for", "from", "by", "with", "at", "as", "is", "are",
    "was", "be", "it", "its", "this", "that", "these", "those", "our", "we", "their", "about", "into",
}


class QueryError(ValueError):
    pass


def clause_to_lucene(c: dict) -> str | None:
    value = str(c.get("value") or "").strip().replace('"', "")
    if not value:
        return None
    kind = c.get("kind") or "term(s)"
    slop = int(c.get("slop") or 0)
    boost = float(c.get("boost") or 1.0)

    if kind == "phrase":
        expr = f'"{value}"' + (f"~{slop}" if slop > 0 else "")
    elif kind == "phrase prefix":
        if len(value.split()) < 2:
            raise QueryError(
                f"Phrase prefix '{value}' needs at least two words "
                '(e.g. "artificial intel") — single-term wildcards are not supported.'
            )
        expr = f'"{value}"*'
    else:
        expr = value

    prefix = OCCUR_PREFIX.get(c.get("occur") or "SHOULD", "")
    if kind == "term(s)" and len(value.split()) > 1 and (prefix or boost != 1.0):
        expr = f"({value})"
    if boost != 1.0:
        expr += f"^{boost:g}"
    return prefix + expr


def clauses_to_lucene(clauses: list[dict]) -> str:
    parts, positive = [], 0
    for c in clauses:
        expr = clause_to_lucene(c)
        if expr is None:
            continue
        parts.append(expr)
        if c.get("occur") != "MUST NOT":
            positive += 1
    if not parts:
        return ""
    if positive == 0:
        raise QueryError("Add at least one SHOULD or MUST clause — a query of only exclusions is rejected.")
    return f"{TEXT_FIELD}:({' '.join(parts)})"


def build_filter(state: dict) -> dict | None:
    conds = []

    text_conds = []
    for f in state.get("text_filters") or []:
        value = str(f.get("value") or "").strip()
        op = f.get("op")
        if not value or op not in MATCH_OPS:
            continue
        cond = {TEXT_FIELD: {op: value}}
        text_conds.append({"$not": cond} if f.get("negate") else cond)
    if len(text_conds) > 1 and state.get("text_filter_join") == "any":
        conds.append({"$or": text_conds})
    else:
        conds.extend(text_conds)

    if state.get("tickers"):
        conds.append({"ticker": {"$in": list(state["tickers"])}})
    if state.get("exclude_tickers"):
        conds.append({"ticker": {"$nin": list(state["exclude_tickers"])}})
    if state.get("year_range"):
        lo, hi = state["year_range"]
        conds.append({"year": {"$gte": lo, "$lte": hi}})
    if state.get("chunk_range"):
        lo, hi = state["chunk_range"]
        conds.append({"chunk_index": {"$gte": lo, "$lte": hi}})

    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def build_score_by(state: dict) -> list[dict]:
    if state["scoring"] == "text":
        query = (state.get("bm25") or "").strip()
        if not query:
            raise QueryError("Enter BM25 keywords.")
        return [{"type": "text", "field": TEXT_FIELD, "query": query}]

    if state.get("lucene_mode") == "raw":
        query = (state.get("raw") or "").strip()
    else:
        query = clauses_to_lucene(state.get("clauses") or [])
    if not query:
        raise QueryError("The Lucene query is empty.")
    return [{"type": "query_string", "query": query}]


def build_request(state: dict, top_k: int, include_fields: list[str]) -> dict:
    req = {
        "namespace": "__default__",
        "top_k": top_k,
        "score_by": build_score_by(state),
        "include_fields": include_fields,
    }
    filt = build_filter(state)
    if filt:
        req["filter"] = filt
    return req


def to_python(req: dict) -> str:
    lines = ["resp = idx.documents.search("]
    for key, value in req.items():
        rendered = json.dumps(value, indent=4).replace("\n", "\n    ")
        lines.append(f"    {key}={rendered},")
    lines.append(")")
    return "\n".join(lines)


def highlight_terms(state: dict) -> list[str]:
    sources = []
    if state["scoring"] == "text":
        sources.append(state.get("bm25") or "")
    elif state.get("lucene_mode") == "raw":
        raw = state.get("raw") or ""
        raw = re.sub(r"-\"[^\"]*\"|-\w+|NOT\s+\w+:\([^)]*\)|NOT\s+\"[^\"]*\"|NOT\s+\w+", " ", raw)
        sources.append(raw)
    else:
        sources += [str(c.get("value") or "") for c in state.get("clauses") or [] if c.get("occur") != "MUST NOT"]
    sources += [str(f.get("value") or "") for f in state.get("text_filters") or [] if not f.get("negate")]

    terms = set()
    for s in sources:
        for word in re.findall(r"[A-Za-z][A-Za-z0-9']+", s):
            if word.lower() not in LUCENE_KEYWORDS | STOP_WORDS:
                terms.add(word.lower())
    return sorted(terms, key=len, reverse=True)


def _clause(value, occur="SHOULD", kind="term(s)", slop=0, boost=1.0):
    return {"occur": occur, "kind": kind, "value": value, "slop": slop, "boost": boost}


def _tf(op, value, negate=False):
    return {"op": op, "value": value, "negate": negate}


BLANK = {
    "scoring": "query_string",
    "bm25": "",
    "lucene_mode": "clauses",
    "clauses": [_clause("")],
    "raw": "",
    "text_filters": [],
    "text_filter_join": "all",
    "tickers": [],
    "exclude_tickers": [],
    "year_range": None,
    "chunk_range": None,
}


def _example(**overrides) -> dict:
    return {**BLANK, **overrides}


EXAMPLES = [
    {
        "name": "Supply-chain shortages, excluding COVID",
        "shows": "Boolean nesting — AND / OR / NOT with a phrase inside a group",
        "description": "Chunks that talk about a shortage of either the supply chain or semiconductors, "
        "but never mention COVID. Parentheses group the OR before the AND.",
        "state": _example(
            lucene_mode="raw",
            raw='text:(("supply chain" OR semiconductor) AND shortage) NOT text:(covid)',
        ),
    },
    {
        "name": "Cyber incidents (required / excluded / boosted)",
        "shows": "Clause builder — MUST (+), MUST NOT (−), SHOULD with a boost",
        "description": "Every result must mention cybersecurity; ransomware mentions count triple, "
        "breach is a nice-to-have, and chunks about insurance are dropped.",
        "state": _example(
            clauses=[
                _clause("cybersecurity", occur="MUST"),
                _clause("ransomware", boost=3.0),
                _clause("breach"),
                _clause("insurance", occur="MUST NOT"),
            ],
        ),
    },
    {
        "name": "Rising interest rates (proximity)",
        "shows": 'Phrase slop — "interest rates increase"~5',
        "description": "The three words must appear within 5 positions of each other, so "
        '"increases in interest rates" and "interest rates may continue to increase" both match. '
        "Stemming lets increase match increases/increased.",
        "state": _example(
            clauses=[_clause("interest rates increase", kind="phrase", slop=5)],
        ),
    },
    {
        "name": "China trade & tariffs (term boost)",
        "shows": "Term boosting — tariffs^3 outweighs trade and china",
        "description": "BM25 over three terms, but tariffs is weighted 3×. Compare with the boost "
        "removed to see ranking shift toward generic trade/China text.",
        "state": _example(
            clauses=[
                _clause("tariffs", boost=3.0),
                _clause("trade"),
                _clause("china"),
            ],
        ),
    },
    {
        "name": "AI mentions (phrase prefix)",
        "shows": 'Phrase prefix — "artificial intel"* matches intelligence / intelligent',
        "description": "The last word is matched as a prefix. Combined with a required cloud term "
        "and restricted to 2022+ filings. Note: prefix matches all share a constant score.",
        "state": _example(
            clauses=[
                _clause("artificial intel", occur="MUST", kind="phrase prefix"),
                _clause("cloud", occur="MUST"),
            ],
            year_range=(2022, 2024),
        ),
    },
    {
        "name": "EV batteries at Ford & GM, 2022+",
        "shows": "BM25 ranking + $match_phrase hard filter + metadata $in / range",
        "description": 'Plain BM25 over battery terms, restricted to chunks containing the exact phrase '
        '"electric vehicles", from Ford or GM, filed 2022–2024.',
        "state": _example(
            scoring="text",
            bm25="battery cells charging range",
            text_filters=[_tf("$match_phrase", "electric vehicles")],
            tickers=["f", "gm"],
            year_range=(2022, 2024),
        ),
    },
    {
        "name": "Cloud growth, no pandemic talk, tech only",
        "shows": "$not + $match_any exclusion filter, ticker $nin",
        "description": "Ranks on cloud revenue growth but excludes any chunk mentioning COVID or "
        "pandemic, and excludes the automakers entirely.",
        "state": _example(
            clauses=[_clause("cloud revenue growth")],
            text_filters=[_tf("$match_any", "covid pandemic", negate=True)],
            exclude_tickers=["f", "gm"],
        ),
    },
    {
        "name": "Regulators: EC or DOJ",
        "shows": "$or across two $match_phrase filters",
        "description": "Antitrust / litigation language, limited to chunks that name either the "
        "European Commission or the Department of Justice.",
        "state": _example(
            clauses=[_clause("antitrust", boost=2.0), _clause("litigation"), _clause("investigation")],
            text_filters=[
                _tf("$match_phrase", "european commission"),
                _tf("$match_phrase", "department of justice"),
            ],
            text_filter_join="any",
        ),
    },
    {
        "name": "Inflation & input costs, 2021–2023",
        "shows": "$match_all (all tokens, any order) + year range",
        "description": "Ranks on inflation; every result must contain raw, materials and costs somewhere "
        "in the chunk (any order), from the 2021–2023 filings.",
        "state": _example(
            scoring="text",
            bm25="inflation inflationary pressures",
            text_filters=[_tf("$match_all", "raw materials costs")],
            year_range=(2021, 2023),
        ),
    },
    {
        "name": "Buybacks vs. dividends (required groups)",
        "shows": "Required OR-groups — +(a OR b) +(c OR d)",
        "description": "Each result must contain a buyback term AND a dividend term; the phrase "
        '"share repurchase" is boosted so it outranks bare "repurchase".',
        "state": _example(
            lucene_mode="raw",
            raw='text:(+("share repurchase"^2 OR buyback OR repurchase) +(dividend OR dividends))',
        ),
    },
]


MODE_EXAMPLES = {
    "fulltext": [
        {
            "name": "Buybacks & dividends",
            "description": "BM25 token-OR: chunks with more (and rarer) of these terms rank higher.",
            "values": {"ft_query": "share repurchase program dividends"},
        },
        {
            "name": "Chip shortage hits production",
            "description": "Surfaces the automakers' 2021–2022 semiconductor supply disclosures.",
            "values": {"ft_query": "semiconductor shortage production"},
        },
        {
            "name": "FX risk",
            "description": "Market-risk language about foreign currency exposure.",
            "values": {"ft_query": "foreign currency exchange rate risk"},
        },
    ],
    "semantic": [
        {
            "name": "Interest-rate exposure (question)",
            "description": "A natural-language question — dense search matches meaning, not exact words.",
            "values": {"sem_query": "How is the company exposed to rising interest rates?"},
        },
        {
            "name": "Supplier concentration in Asia",
            "description": "Filings rarely use this wording; embeddings find single-source and "
            "outsourcing-partner risk disclosures anyway.",
            "values": {"sem_query": "Concerns about dependence on a small number of suppliers in Asia"},
        },
        {
            "name": "AI & data-center buildout",
            "description": "Conceptual query that spans capex, infrastructure and AI discussion.",
            "values": {"sem_query": "Investments in artificial intelligence and data center capacity"},
        },
    ],
    "hybrid": [
        {
            "name": "Cloud growth that names Azure",
            "description": "Semantic ranking on cloud growth, but only chunks containing \"Azure\".",
            "values": {"hy_sem": "growth of cloud infrastructure services", "hy_txt": "Azure"},
        },
        {
            "name": "EV transition that names Ultium",
            "description": "GM's battery platform as a hard keyword; ranking by the EV-transition concept.",
            "values": {"hy_sem": "transition to electric vehicles and battery supply", "hy_txt": "Ultium"},
        },
        {
            "name": "Competition probes by the EC",
            "description": "Regulatory-investigation meaning, restricted to chunks mentioning European Commission.",
            "values": {
                "hy_sem": "regulatory investigations into competition practices",
                "hy_txt": "European Commission",
            },
        },
    ],
    "compare": [
        {
            "name": "Interest rates vs consumer demand",
            "description": "Same words on both sides: BM25 rewards literal term hits, dense rewards the idea.",
            "values": {"c_query": "risks from rising interest rates on consumer demand", "c_source": "same"},
        },
        {
            "name": "Paraphrase: overseas manufacturing partners",
            "description": "Wording filings don't use verbatim — expect low overlap and low keyword "
            "coverage on the dense side, where it still finds outsourcing-risk passages.",
            "values": {
                "c_query": "the company depends on a few manufacturing partners overseas",
                "c_source": "same",
            },
        },
        {
            "name": "Lucene builder query vs dense",
            "description": "Loads the \"Supply-chain shortages, excluding COVID\" builder query as the "
            "full-text side against a dense search for the same topic.",
            "values": {"c_query": "semiconductor chip shortage impact on production", "c_source": "builder"},
            "builder_example": "Supply-chain shortages, excluding COVID",
        },
    ],
}
