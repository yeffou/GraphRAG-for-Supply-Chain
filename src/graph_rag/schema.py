"""Domain schema, corpus-driven concept mining, and deterministic extraction."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


GRAPH_SCHEMA_VERSION = "graph_entity_relation_v3"

STOPWORDS = {
    "a",
    "about",
    "above",
    "across",
    "after",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "between",
    "both",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "during",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "less",
    "may",
    "more",
    "most",
    "not",
    "of",
    "on",
    "only",
    "or",
    "other",
    "our",
    "over",
    "same",
    "should",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "under",
    "use",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "why",
    "will",
    "with",
    "within",
}

BLOCKED_MINED_TOKENS = {
    "annex",
    "appendix",
    "chapter",
    "doi",
    "figure",
    "https",
    "http",
    "lin",
    "note",
    "notes",
    "number",
    "page",
    "paper",
    "pp",
    "report",
    "review",
    "table",
    "vol",
    "www",
    "zero",
    "one",
    "two",
}

BLOCKED_MINED_PHRASES = {
    "as well",
    "as well as",
    "based on",
    "number of",
    "oecd supply chain",
    "oecd supply chain resilience",
    "resilience review",
    "resilience review oecd",
    "supply chain resilience review",
}

DOMAIN_CUE_PREFIXES = (
    "adapt",
    "agil",
    "align",
    "barrier",
    "buyer",
    "collabor",
    "concentr",
    "constraint",
    "cooper",
    "coord",
    "custom",
    "digit",
    "disrupt",
    "divers",
    "efficien",
    "export",
    "facilitat",
    "financial",
    "flexib",
    "foreign",
    "hedg",
    "indic",
    "industr",
    "inform",
    "inclus",
    "insur",
    "interoper",
    "interven",
    "inventor",
    "joint",
    "logistic",
    "manufactur",
    "market",
    "material",
    "mitig",
    "outsour",
    "prepared",
    "product",
    "productiv",
    "raw",
    "real",
    "regulator",
    "resilien",
    "respons",
    "restrict",
    "risk",
    "service",
    "shortage",
    "shock",
    "subcontract",
    "supplier",
    "supply",
    "sustain",
    "telecom",
    "trade",
    "transaction",
    "transport",
    "transparen",
    "value",
    "veloc",
    "visib",
    "vulnerab",
    "warning",
)

GENERIC_BASE_CANONICALS = {
    "Supply Chain",
    "Resilience",
    "OECD",
    "WTO",
    "UNCTAD",
    "World Bank",
    "Global Value Chain",
    "International Trade",
    "Shock",
    "Preparedness",
}

ENTITY_TYPE_WEIGHTS = {
    "strategy": 3.0,
    "policy": 2.9,
    "capability": 2.6,
    "risk": 2.4,
    "actor": 1.9,
    "sector": 1.9,
    "system": 1.1,
    "organization": 0.7,
    "location": 0.5,
}

RELATION_TYPE_WEIGHTS = {
    "MENTIONED_IN": 1.0,
    "STRENGTHENS": 3.2,
    "IMPROVES": 3.0,
    "MITIGATES": 2.9,
    "TRADE_OFF_WITH": 2.8,
    "ENABLES": 2.7,
    "EXPOSES": 2.6,
    "CONSTRAINS": 2.5,
    "AFFECTS": 2.1,
    "APPLIES_TO": 1.2,
    "CO_OCCURS_WITH": 0.3,
}

IMPROVEMENT_TRIGGERS = (
    "improve",
    "improves",
    "improved",
    "improving",
    "strengthen",
    "strengthens",
    "strengthened",
    "strengthening",
    "enhance",
    "enhances",
    "enhanced",
    "enhancing",
    "boost",
    "boosts",
    "support resilience",
    "increase resilience",
    "build resilience",
)

ENABLEMENT_TRIGGERS = (
    "allow",
    "allows",
    "enable",
    "enables",
    "enabled",
    "facilitate",
    "facilitates",
    "facilitated",
    "help",
    "helps",
    "make it possible",
    "support",
    "supports",
)

MITIGATION_TRIGGERS = (
    "avoid",
    "buffer",
    "cope with",
    "de-risk",
    "de-risking",
    "hedge",
    "mitigate",
    "mitigates",
    "prepare",
    "protect",
    "recover",
    "reduce",
    "reduces",
    "surmount",
    "withstand",
)

CONSTRAINT_TRIGGERS = (
    "barrier",
    "barriers",
    "burdensome",
    "constrain",
    "constrains",
    "constraint",
    "constraints",
    "cost",
    "costs",
    "limit",
    "limits",
    "restriction",
    "restrictions",
)

EXPOSURE_TRIGGERS = (
    "concentration",
    "concentrated",
    "dependence",
    "dependency",
    "exposure",
    "exposed",
    "vulnerable",
    "vulnerability",
)

AFFECT_TRIGGERS = (
    "affect",
    "bottleneck",
    "bottlenecks",
    "delay",
    "delays",
    "disruption",
    "disruptions",
    "expose",
    "impact",
    "loss",
    "pressure",
    "shortage",
    "shortages",
    "shock",
    "shocks",
)

TRADE_OFF_TRIGGERS = (
    "balancing",
    "balance",
    "interplay between",
    "inverse relationship",
    "tension between",
    "trade off",
    "trade-off",
    "trade-offs",
    "trade offs",
    "tradeoffs",
)

NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
TOKEN_RE = re.compile(r"[a-z][a-z0-9-]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


@dataclass(frozen=True)
class EntitySpec:
    entity_type: str
    canonical_name: str
    aliases: tuple[str, ...]
    source: str = "base"
    generic_hint: bool = False


class SavedEntitySpecRecord(BaseModel):
    """Serializable entity spec record stored with the graph index."""

    model_config = ConfigDict(extra="forbid")

    entity_type: str
    canonical_name: str
    aliases: list[str]
    source: str
    generic_hint: bool = False


class EntityMention(BaseModel):
    """Entity mention extracted from a chunk or query."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    canonical_name: str
    entity_type: str
    matched_alias: str
    source: str = "base"
    is_generic: bool = False


class ExtractedRelation(BaseModel):
    """Typed entity relation inferred within one sentence."""

    model_config = ConfigDict(extra="forbid")

    source_entity_id: str
    source_name: str
    source_type: str
    relation_type: str
    target_entity_id: str
    target_name: str
    target_type: str
    sentence_index: int = Field(ge=0)
    trigger: str | None = None
    evidence_text: str


class SentenceExtractionRecord(BaseModel):
    """Sentence-level evidence record for graph indexing and scoring."""

    model_config = ConfigDict(extra="forbid")

    sentence_id: str
    sentence_index: int = Field(ge=0)
    text: str
    extracted_entities: list[EntityMention]
    extracted_relations: list[ExtractedRelation]


BASE_ENTITY_SPECS: tuple[EntitySpec, ...] = (
    EntitySpec("organization", "WTO", ("wto", "world trade organization"), generic_hint=True),
    EntitySpec("organization", "OECD", ("oecd", "organisation for economic co-operation and development"), generic_hint=True),
    EntitySpec("organization", "UNCTAD", ("unctad", "united nations conference on trade and development"), generic_hint=True),
    EntitySpec("organization", "World Bank", ("world bank",), generic_hint=True),
    EntitySpec("organization", "WHO", ("who", "world health organization")),
    EntitySpec("organization", "IMF", ("imf", "international monetary fund")),
    EntitySpec("organization", "FAO", ("fao", "food and agriculture organization")),
    EntitySpec("organization", "UN", ("united nations", "un")),
    EntitySpec("organization", "UNDRR", ("undrr", "united nations office for disaster risk reduction")),
    EntitySpec("organization", "WCO", ("wco", "world customs organization")),
    EntitySpec("organization", "European Union", ("european union", "eu")),
    EntitySpec("organization", "African Continental Free Trade Area", ("african continental free trade area", "afcfta")),
    EntitySpec("location", "Africa", ("africa", "african countries")),
    EntitySpec("location", "Asia", ("asia",)),
    EntitySpec("location", "Europe", ("europe",)),
    EntitySpec("location", "United States", ("united states", "u s", "usa")),
    EntitySpec("location", "China", ("china",)),
    EntitySpec("location", "Nigeria", ("nigeria",)),
    EntitySpec("location", "Democratic Republic of the Congo", ("democratic republic of the congo", "dr congo")),
    EntitySpec("location", "United Kingdom", ("united kingdom",)),
    EntitySpec("location", "Ukraine", ("ukraine", "war in ukraine")),
    EntitySpec("risk", "Pandemic", ("pandemic", "pandemics")),
    EntitySpec("risk", "COVID-19", ("covid 19", "covid-19", "coronavirus disease")),
    EntitySpec("risk", "Climate Change", ("climate change",)),
    EntitySpec("risk", "Natural Disaster", ("natural disaster", "natural disasters")),
    EntitySpec("risk", "Flood", ("flood", "floods", "flooding")),
    EntitySpec("risk", "Drought", ("drought", "droughts")),
    EntitySpec("risk", "Cyclone", ("cyclone", "cyclones")),
    EntitySpec("risk", "Earthquake", ("earthquake", "earthquakes")),
    EntitySpec("risk", "Cyberattack", ("cyberattack", "cyberattacks", "cyber-attack", "cyber-attacks")),
    EntitySpec("risk", "Trade Dispute", ("trade dispute", "trade disputes")),
    EntitySpec("risk", "Supply Chain Disruption", ("supply chain disruption", "supply chain disruptions", "supply disruption", "supply disruptions", "business interruption", "business interruptions", "disruption", "disruptions")),
    EntitySpec("risk", "Bottleneck", ("bottleneck", "bottlenecks")),
    EntitySpec("risk", "Shortage", ("shortage", "shortages")),
    EntitySpec("risk", "Shock", ("shock", "shocks"), generic_hint=True),
    EntitySpec("risk", "Vulnerability", ("vulnerability", "vulnerabilities", "vulnerable")),
    EntitySpec("risk", "Concentration", ("concentration", "concentrated", "high concentration")),
    EntitySpec("strategy", "Diversification", ("diversification", "diversify", "diversifies", "diversified", "diversifying", "supply chain diversification", "supply base diversification", "supplier diversification", "trade diversification", "multi-sourcing", "multiple sourcing")),
    EntitySpec("strategy", "Reshoring", ("reshoring", "re-shoring", "reshore")),
    EntitySpec("strategy", "Nearshoring", ("nearshoring", "near-shoring", "nearshore")),
    EntitySpec("strategy", "Collaboration", ("collaboration", "collaborative")),
    EntitySpec("strategy", "Coordination", ("coordination", "co-ordination")),
    EntitySpec("strategy", "Information Sharing", ("information sharing", "information-sharing", "sharing information")),
    EntitySpec("strategy", "International Cooperation", ("international cooperation", "trade cooperation", "global cooperation")),
    EntitySpec("strategy", "Regional Integration", ("regional integration",)),
    EntitySpec("strategy", "Inventory Stocks", ("inventory stocks", "inventory stock", "stockpiling", "buffer stocks", "safety stocks")),
    EntitySpec("strategy", "Just-in-Time", ("just in time", "just-in-time")),
    EntitySpec("strategy", "Just-in-Case", ("just in case", "just-in-case")),
    EntitySpec("strategy", "Risk Management", ("risk management",), generic_hint=True),
    EntitySpec("strategy", "Early Warning", ("early warning", "early warning signs", "weather forecasting")),
    EntitySpec("strategy", "Outsourcing", ("outsourcing",)),
    EntitySpec("strategy", "Subcontracting", ("subcontracting",)),
    EntitySpec("strategy", "Joint Ventures", ("joint ventures", "joint venture")),
    EntitySpec("strategy", "Hedging", ("hedging", "hedge")),
    EntitySpec("strategy", "Insurance", ("insurance",)),
    EntitySpec("capability", "Resilience", ("resilience", "resilient", "supply chain resilience", "economic resilience"), generic_hint=True),
    EntitySpec("capability", "Efficiency", ("efficiency", "efficient", "economic efficiency", "cost efficiency")),
    EntitySpec("capability", "Preparedness", ("preparedness", "prepared"), generic_hint=True),
    EntitySpec("capability", "Flexibility", ("flexibility", "flexible")),
    EntitySpec("capability", "Agility", ("agility", "agile")),
    EntitySpec("capability", "Adaptability", ("adaptability", "adaptable")),
    EntitySpec("capability", "Alignment", ("alignment", "aligned")),
    EntitySpec("capability", "Visibility", ("visibility", "end-to-end supply chain visibility")),
    EntitySpec("capability", "Transparency", ("transparency", "transparent")),
    EntitySpec("capability", "Velocity", ("velocity",)),
    EntitySpec("capability", "Recovery", ("recovery", "recover")),
    EntitySpec("capability", "Redundancy", ("redundancy", "redundant")),
    EntitySpec("capability", "Responsiveness", ("responsiveness", "responsive", "culture of responsiveness")),
    EntitySpec("policy", "Trade Facilitation", ("trade facilitation", "trade facilitation policies", "streamlining border procedures", "customs laws")),
    EntitySpec("policy", "Export Restrictions", ("export restrictions", "export restriction", "restriction measures")),
    EntitySpec("policy", "Market Access Barriers", ("market access barriers", "market access barrier")),
    EntitySpec("policy", "Regulatory Interoperability", ("regulatory interoperability",)),
    EntitySpec("policy", "Intervention Thresholds", ("intervention thresholds", "intervention threshold")),
    EntitySpec("policy", "Sustainability Regulations", ("sustainability regulations", "sustainability-oriented regulations", "supply-chain regulations")),
    EntitySpec("system", "Supply Chain", ("supply chain", "supply chains"), generic_hint=True),
    EntitySpec("system", "Global Supply Chain", ("global supply chain", "global supply chains")),
    EntitySpec("system", "Global Value Chain", ("global value chain", "global value chains", "gvc", "gvcs"), generic_hint=True),
    EntitySpec("system", "Transport System", ("transport system", "transport systems")),
    EntitySpec("system", "Logistics", ("logistics",)),
    EntitySpec("system", "Port Infrastructure", ("port infrastructure", "ports", "port")),
    EntitySpec("system", "Production Network", ("production network", "production networks")),
    EntitySpec("system", "International Trade", ("international trade", "multilateral trading system"), generic_hint=True),
    EntitySpec("system", "Trade in Services", ("trade in services", "services trade")),
    EntitySpec("sector", "Semiconductors", ("semiconductors", "semiconductor")),
    EntitySpec("sector", "Vaccines", ("vaccines", "vaccine")),
    EntitySpec("sector", "Medical Products", ("medical products", "medical product")),
    EntitySpec("sector", "Medical Supplies", ("medical supplies", "medical supply")),
    EntitySpec("sector", "Pharmaceuticals", ("pharmaceuticals", "pharmaceutical")),
    EntitySpec("sector", "Personal Protective Equipment", ("personal protective equipment", "ppe", "face masks", "textile face masks")),
    EntitySpec("sector", "Food", ("food", "agricultural products")),
    EntitySpec("sector", "Electronics", ("electronics", "electronic goods", "computers and electronics")),
    EntitySpec("sector", "Transport Equipment", ("transport equipment", "transportation equipment")),
    EntitySpec("sector", "Transport Services", ("transport services", "transport service")),
    EntitySpec("sector", "Logistics Services", ("logistics services", "logistics service")),
    EntitySpec("sector", "Financial Services", ("financial services", "financial service")),
    EntitySpec("sector", "Telecommunications Services", ("telecommunications services", "telecommunications service", "telecommunications")),
    EntitySpec("sector", "Critical Raw Materials", ("critical raw materials", "critical raw material")),
    EntitySpec("sector", "Raw Materials", ("raw materials", "raw material")),
    EntitySpec("sector", "Strategic Manufacturing", ("strategic manufacturing", "strategic manufacturing industries")),
    EntitySpec("sector", "Communication Equipment", ("communication equipment",)),
    EntitySpec("sector", "Medical Devices", ("medical devices", "medical device")),
    EntitySpec("sector", "Copper", ("copper",)),
    EntitySpec("sector", "Cobalt", ("cobalt", "cobalt oxides")),
    EntitySpec("sector", "Automotive", ("automotive", "electric vehicles")),
    EntitySpec("actor", "Suppliers", ("suppliers", "supplier")),
    EntitySpec("actor", "Foreign Suppliers", ("foreign suppliers", "foreign supplier")),
    EntitySpec("actor", "New Suppliers", ("new suppliers", "new supplier")),
    EntitySpec("actor", "New Buyers", ("new buyers", "new buyer")),
    EntitySpec("actor", "Buyers", ("buyers", "buyer")),
    EntitySpec("actor", "Firms", ("firms", "firm")),
    EntitySpec("actor", "Governments", ("governments", "government")),
)

DOMAIN_TAG_TO_ENTITY = {
    "supply_chain": ("system", "Supply Chain"),
    "trade": ("system", "International Trade"),
    "resilience": ("capability", "Resilience"),
    "global_value_chains": ("system", "Global Value Chain"),
    "transport": ("system", "Transport System"),
    "infrastructure": ("system", "Port Infrastructure"),
    "preparedness": ("capability", "Preparedness"),
    "collaboration": ("strategy", "Collaboration"),
    "coordination": ("strategy", "Coordination"),
    "management": ("strategy", "Risk Management"),
}


def slugify(value: str) -> str:
    """Create a filesystem-safe slug."""

    return NORMALIZE_RE.sub("_", value.lower()).strip("_")


def entity_id_for(spec: EntitySpec) -> str:
    return f"entity::{spec.entity_type}::{slugify(spec.canonical_name)}"


def base_entity_specs() -> tuple[EntitySpec, ...]:
    return BASE_ENTITY_SPECS


def default_entity_spec_by_id() -> dict[str, EntitySpec]:
    return {entity_id_for(spec): spec for spec in BASE_ENTITY_SPECS}


def normalize_text_for_matching(text: str) -> str:
    """Normalize text for deterministic alias matching."""

    lowered = text.lower()
    replaced = NORMALIZE_RE.sub(" ", lowered)
    return f" {re.sub(r'\\s+', ' ', replaced).strip()} "


def pretty_canonical_name(phrase: str) -> str:
    return " ".join(part.capitalize() for part in phrase.split())


def serialize_entity_specs(specs: tuple[EntitySpec, ...], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            record = SavedEntitySpecRecord(
                entity_type=spec.entity_type,
                canonical_name=spec.canonical_name,
                aliases=list(spec.aliases),
                source=spec.source,
                generic_hint=spec.generic_hint,
            )
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")


def load_entity_specs(path: Path) -> tuple[EntitySpec, ...]:
    specs: list[EntitySpec] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            record = SavedEntitySpecRecord.model_validate(payload)
            specs.append(
                EntitySpec(
                    entity_type=record.entity_type,
                    canonical_name=record.canonical_name,
                    aliases=tuple(record.aliases),
                    source=record.source,
                    generic_hint=record.generic_hint,
                )
            )
    return tuple(specs)


def build_corpus_entity_specs(chunks, max_dynamic_specs: int = 160) -> tuple[EntitySpec, ...]:
    """Build the active entity vocabulary from the corpus plus core ontology."""

    base_specs = base_entity_specs()
    base_aliases = {
        normalize_text_for_matching(alias).strip()
        for spec in base_specs
        for alias in spec.aliases + (spec.canonical_name.lower(),)
    }
    document_frequency: Counter[str] = Counter()

    for chunk in chunks:
        normalized_tokens = TOKEN_RE.findall(chunk.text.lower())
        candidates_in_chunk: set[str] = set()
        for n in (1, 2, 3, 4):
            for index in range(len(normalized_tokens) - n + 1):
                phrase = " ".join(normalized_tokens[index : index + n])
                if not _is_valid_mined_candidate(phrase):
                    continue
                if normalize_text_for_matching(phrase).strip() in base_aliases:
                    continue
                entity_type = infer_entity_type_for_phrase(phrase)
                if entity_type is None:
                    continue
                if not _passes_frequency_threshold(phrase, entity_type):
                    continue
                candidates_in_chunk.add(phrase)
        for candidate in candidates_in_chunk:
            document_frequency[candidate] += 1

    dynamic_specs: list[tuple[float, EntitySpec]] = []
    seen_entity_ids = {entity_id_for(spec) for spec in base_specs}
    for phrase, chunk_count in document_frequency.items():
        entity_type = infer_entity_type_for_phrase(phrase)
        if entity_type is None:
            continue
        if not _meets_dynamic_threshold(phrase, chunk_count):
            continue
        aliases = generate_phrase_aliases(phrase)
        spec = EntitySpec(
            entity_type=entity_type,
            canonical_name=pretty_canonical_name(phrase),
            aliases=aliases,
            source="mined",
            generic_hint=False,
        )
        entity_id = entity_id_for(spec)
        if entity_id in seen_entity_ids:
            continue
        seen_entity_ids.add(entity_id)
        score = chunk_count * (1.0 + 0.20 * (len(phrase.split()) - 1))
        dynamic_specs.append((score, spec))

    dynamic_specs.sort(
        key=lambda item: (item[0], len(item[1].canonical_name)),
        reverse=True,
    )
    trimmed_dynamic = tuple(spec for _, spec in dynamic_specs[:max_dynamic_specs])
    return tuple(sorted(base_specs + trimmed_dynamic, key=lambda spec: (spec.entity_type, spec.canonical_name)))


def extract_entities(
    text: str,
    domain_tags: list[str] | None = None,
    allow_domain_tag_backfill: bool = True,
    entity_specs: tuple[EntitySpec, ...] | None = None,
) -> list[EntityMention]:
    """Extract typed entities using the provided active vocabulary."""

    active_specs = entity_specs or base_entity_specs()
    normalized_text = normalize_text_for_matching(text)
    found_mentions: dict[str, EntityMention] = {}

    for spec in sorted(
        active_specs,
        key=lambda item: max(len(alias) for alias in item.aliases),
        reverse=True,
    ):
        for alias in spec.aliases:
            alias_norm = normalize_text_for_matching(alias)
            if alias_norm in normalized_text:
                entity_id = entity_id_for(spec)
                found_mentions.setdefault(
                    entity_id,
                    EntityMention(
                        entity_id=entity_id,
                        canonical_name=spec.canonical_name,
                        entity_type=spec.entity_type,
                        matched_alias=alias,
                        source=spec.source,
                        is_generic=spec.generic_hint,
                    ),
                )
                break

    if allow_domain_tag_backfill and domain_tags:
        spec_by_id = {
            entity_id_for(spec): spec for spec in active_specs
        }
        for tag in domain_tags:
            mapping = DOMAIN_TAG_TO_ENTITY.get(tag)
            if not mapping:
                continue
            entity_type, canonical_name = mapping
            entity_id = f"entity::{entity_type}::{slugify(canonical_name)}"
            spec = spec_by_id.get(entity_id)
            if entity_id not in found_mentions:
                found_mentions[entity_id] = EntityMention(
                    entity_id=entity_id,
                    canonical_name=canonical_name,
                    entity_type=entity_type,
                    matched_alias=tag,
                    source=spec.source if spec else "domain_tag",
                    is_generic=spec.generic_hint if spec else False,
                )

    return sorted(found_mentions.values(), key=lambda mention: mention.entity_id)


def extract_sentence_records(
    text: str,
    entity_specs: tuple[EntitySpec, ...] | None = None,
) -> list[SentenceExtractionRecord]:
    """Extract sentence-level entities and relations for one chunk."""

    records: list[SentenceExtractionRecord] = []
    for sentence_index, sentence_text in enumerate(split_text_into_sentences(text)):
        sentence_entities = extract_entities(
            text=sentence_text,
            domain_tags=None,
            allow_domain_tag_backfill=False,
            entity_specs=entity_specs,
        )
        sentence_relations = infer_sentence_relations(
            sentence_text=sentence_text,
            sentence_entities=sentence_entities,
            sentence_index=sentence_index,
        )
        records.append(
            SentenceExtractionRecord(
                sentence_id=f"sentence::{sentence_index}",
                sentence_index=sentence_index,
                text=sentence_text,
                extracted_entities=sentence_entities,
                extracted_relations=sentence_relations,
            )
        )
    return records


def infer_relations(
    text: str,
    mentions: list[EntityMention] | None = None,
    entity_specs: tuple[EntitySpec, ...] | None = None,
) -> list[ExtractedRelation]:
    """Infer typed relations across all sentences in one chunk."""

    del mentions
    relations: list[ExtractedRelation] = []
    for sentence_record in extract_sentence_records(text=text, entity_specs=entity_specs):
        relations.extend(sentence_record.extracted_relations)
    return relations


def infer_sentence_relations(
    sentence_text: str,
    sentence_entities: list[EntityMention],
    sentence_index: int,
) -> list[ExtractedRelation]:
    """Infer typed relations inside one sentence."""

    if len(sentence_entities) < 2:
        return []

    normalized_sentence = normalize_text_for_matching(sentence_text)
    relations: list[ExtractedRelation] = []
    seen_keys: set[tuple[str, str, str]] = set()
    allow_co_occurrence = len(sentence_entities) <= 4

    for index, left in enumerate(sentence_entities):
        for right in sentence_entities[index + 1 :]:
            relation = classify_relation(
                left=left,
                right=right,
                sentence_text=sentence_text,
                normalized_text=normalized_sentence,
                sentence_index=sentence_index,
                allow_co_occurrence=allow_co_occurrence,
            )
            if relation is None:
                continue
            key = (
                relation.source_entity_id,
                relation.relation_type,
                relation.target_entity_id,
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            relations.append(relation)
    return relations


def classify_relation(
    left: EntityMention,
    right: EntityMention,
    sentence_text: str,
    normalized_text: str,
    sentence_index: int,
    allow_co_occurrence: bool,
) -> ExtractedRelation | None:
    """Classify the strongest deterministic relation between two sentence entities."""

    if _is_tradeoff_pair(left, right):
        trigger = _matching_trigger(normalized_text, TRADE_OFF_TRIGGERS)
        if trigger:
            source, target = _sorted_pair(left, right)
            return _relation(source, "TRADE_OFF_WITH", target, sentence_index, trigger, sentence_text)

    strengthen_trigger = _matching_trigger(normalized_text, IMPROVEMENT_TRIGGERS)
    if strengthen_trigger and _is_strengthening_pair(left, right):
        source, target = _directional_pair(left, right, preferred_source_types=("strategy", "policy", "capability"))
        return _relation(source, "STRENGTHENS", target, sentence_index, strengthen_trigger, sentence_text)

    if _is_diversification_resilience_pair(left, right):
        source, target = _directional_pair(left, right, preferred_source_types=("strategy",))
        return _relation(source, "STRENGTHENS", target, sentence_index, "direct_pair", sentence_text)

    enable_trigger = _matching_trigger(normalized_text, ENABLEMENT_TRIGGERS)
    if enable_trigger and _is_enablement_pair(left, right):
        source, target = _directional_pair(left, right, preferred_source_types=("strategy", "policy", "capability"))
        return _relation(source, "ENABLES", target, sentence_index, enable_trigger, sentence_text)

    mitigate_trigger = _matching_trigger(normalized_text, MITIGATION_TRIGGERS)
    if mitigate_trigger and _is_mitigation_pair(left, right):
        source, target = _directional_pair(left, right, preferred_source_types=("strategy", "policy", "capability"))
        return _relation(source, "MITIGATES", target, sentence_index, mitigate_trigger, sentence_text)

    constraint_trigger = _matching_trigger(normalized_text, CONSTRAINT_TRIGGERS)
    if constraint_trigger and _is_constraint_pair(left, right):
        source, target = _directional_pair(left, right, preferred_source_types=("policy", "risk"))
        return _relation(source, "CONSTRAINS", target, sentence_index, constraint_trigger, sentence_text)

    exposure_trigger = _matching_trigger(normalized_text, EXPOSURE_TRIGGERS)
    if exposure_trigger and _is_exposure_pair(left, right):
        source, target = _directional_pair(left, right, preferred_source_types=("risk", "policy", "sector", "actor"))
        return _relation(source, "EXPOSES", target, sentence_index, exposure_trigger, sentence_text)

    affect_trigger = _matching_trigger(normalized_text, AFFECT_TRIGGERS)
    if affect_trigger and _is_affect_pair(left, right):
        source, target = _directional_pair(left, right, preferred_source_types=("risk", "policy"))
        return _relation(source, "AFFECTS", target, sentence_index, affect_trigger, sentence_text)

    if _is_applicability_pair(left, right):
        source, target = _directional_pair(left, right, preferred_source_types=("strategy", "policy", "capability"))
        return _relation(source, "APPLIES_TO", target, sentence_index, None, sentence_text)

    if allow_co_occurrence and _allow_co_occurrence(left, right):
        source, target = _sorted_pair(left, right)
        return _relation(source, "CO_OCCURS_WITH", target, sentence_index, None, sentence_text)

    return None


def split_text_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like spans."""

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]
    return sentences or [cleaned]


def query_terms(text: str) -> set[str]:
    """Extract normalized query terms for lexical tie-breaking."""

    return {
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 2
    }


def infer_entity_type_for_phrase(phrase: str) -> str | None:
    tokens = phrase.split()
    if not tokens:
        return None
    joined = " ".join(tokens)
    if any(token.startswith(("supplier", "buyer", "firm", "government")) for token in tokens):
        return "actor"
    if any(token.startswith(("barrier", "custom", "facilitat", "interoper", "interven", "regulat", "restrict")) for token in tokens):
        return "policy"
    if any(token.startswith(("adapt", "agil", "align", "efficien", "flexib", "prepared", "resilien", "respons", "transparen", "veloc", "visib")) for token in tokens):
        return "capability"
    if any(token.startswith(("collabor", "coord", "divers", "early", "hedg", "insur", "inventor", "joint", "multi", "outsour", "stock", "subcontract")) for token in tokens):
        return "strategy"
    if any(token.startswith(("bottleneck", "climate", "concentr", "cyber", "disrupt", "exposure", "pandemic", "risk", "shock", "shortage", "vulnerab")) for token in tokens):
        return "risk"
    if any(token.startswith(("service", "materials", "material", "manufactur", "pharma", "telecom", "transport", "logistic", "financial", "semiconductor", "vaccine")) for token in tokens):
        return "sector"
    if any(token.startswith(("chain", "logistic", "trade", "transport")) for token in tokens):
        return "system"
    if "critical raw materials" in joined:
        return "sector"
    if "trade in services" in joined:
        return "system"
    return None


def generate_phrase_aliases(phrase: str) -> tuple[str, ...]:
    aliases = {phrase}
    if " " in phrase:
        aliases.add(phrase.replace(" ", "-"))
    if "-" in phrase:
        aliases.add(phrase.replace("-", " "))
    if phrase.endswith("s") and len(phrase) > 4:
        aliases.add(phrase[:-1])
    elif len(phrase) > 4:
        aliases.add(f"{phrase}s")
    return tuple(sorted(alias for alias in aliases if alias))


def _is_valid_mined_candidate(phrase: str) -> bool:
    if phrase in BLOCKED_MINED_PHRASES:
        return False
    tokens = phrase.split()
    if not tokens or len(tokens) > 4:
        return False
    if any(token in BLOCKED_MINED_TOKENS for token in tokens):
        return False
    if tokens[0] in STOPWORDS or tokens[-1] in STOPWORDS:
        return False
    if not any(any(token.startswith(prefix) for prefix in DOMAIN_CUE_PREFIXES) for token in tokens):
        return False
    if any(token.isdigit() for token in tokens):
        return False
    if len(set(tokens)) == 1 and len(tokens) > 1:
        return False
    return True


def _passes_frequency_threshold(phrase: str, entity_type: str | None) -> bool:
    del entity_type
    token_count = len(phrase.split())
    if token_count == 1:
        return False
    return True


def _meets_dynamic_threshold(phrase: str, chunk_count: int) -> bool:
    token_count = len(phrase.split())
    if token_count == 1:
        return chunk_count >= 6
    if token_count == 2:
        return chunk_count >= 3
    return chunk_count >= 2


def _relation(
    source: EntityMention,
    relation_type: str,
    target: EntityMention,
    sentence_index: int,
    trigger: str | None,
    evidence_text: str,
) -> ExtractedRelation:
    return ExtractedRelation(
        source_entity_id=source.entity_id,
        source_name=source.canonical_name,
        source_type=source.entity_type,
        relation_type=relation_type,
        target_entity_id=target.entity_id,
        target_name=target.canonical_name,
        target_type=target.entity_type,
        sentence_index=sentence_index,
        trigger=trigger,
        evidence_text=evidence_text.strip(),
    )


def _directional_pair(
    left: EntityMention,
    right: EntityMention,
    preferred_source_type: str | None = None,
    preferred_source_types: tuple[str, ...] | None = None,
) -> tuple[EntityMention, EntityMention]:
    if preferred_source_type:
        if left.entity_type == preferred_source_type:
            return left, right
        if right.entity_type == preferred_source_type:
            return right, left
    if preferred_source_types:
        left_in = left.entity_type in preferred_source_types
        right_in = right.entity_type in preferred_source_types
        if left_in and not right_in:
            return left, right
        if right_in and not left_in:
            return right, left
    return _sorted_pair(left, right)


def _sorted_pair(left: EntityMention, right: EntityMention) -> tuple[EntityMention, EntityMention]:
    if left.entity_id <= right.entity_id:
        return left, right
    return right, left


def _is_tradeoff_pair(left: EntityMention, right: EntityMention) -> bool:
    return (
        left.entity_type == right.entity_type == "capability"
        or {left.canonical_name, right.canonical_name} == {"Efficiency", "Resilience"}
    )


def _is_strengthening_pair(left: EntityMention, right: EntityMention) -> bool:
    sources = {"strategy", "policy", "capability"}
    targets = {"capability", "system", "actor", "sector"}
    return (
        (left.entity_type in sources and right.entity_type in targets)
        or (right.entity_type in sources and left.entity_type in targets)
    )


def _is_enablement_pair(left: EntityMention, right: EntityMention) -> bool:
    return _is_strengthening_pair(left, right)


def _is_mitigation_pair(left: EntityMention, right: EntityMention) -> bool:
    sources = {"strategy", "policy", "capability"}
    targets = {"risk", "system", "actor", "sector"}
    return (
        (left.entity_type in sources and right.entity_type in targets)
        or (right.entity_type in sources and left.entity_type in targets)
    )


def _is_constraint_pair(left: EntityMention, right: EntityMention) -> bool:
    sources = {"policy", "risk"}
    targets = {"system", "actor", "sector", "capability"}
    return (
        (left.entity_type in sources and right.entity_type in targets)
        or (right.entity_type in sources and left.entity_type in targets)
    )


def _is_exposure_pair(left: EntityMention, right: EntityMention) -> bool:
    sources = {"policy", "risk", "sector", "actor"}
    targets = {"system", "capability", "actor", "sector"}
    return (
        (left.entity_type in sources and right.entity_type in targets)
        or (right.entity_type in sources and left.entity_type in targets)
    )


def _is_affect_pair(left: EntityMention, right: EntityMention) -> bool:
    sources = {"risk", "policy"}
    targets = {"system", "sector", "actor", "capability", "location", "organization"}
    return (
        (left.entity_type in sources and right.entity_type in targets)
        or (right.entity_type in sources and left.entity_type in targets)
    )


def _is_applicability_pair(left: EntityMention, right: EntityMention) -> bool:
    sources = {"strategy", "policy", "capability"}
    targets = {"system", "sector", "actor"}
    return (
        (left.entity_type in sources and right.entity_type in targets)
        or (right.entity_type in sources and left.entity_type in targets)
    )


def _allow_co_occurrence(left: EntityMention, right: EntityMention) -> bool:
    allowed_types = {"strategy", "policy", "capability", "risk", "actor", "sector"}
    if left.entity_type not in allowed_types or right.entity_type not in allowed_types:
        return False
    return not (left.is_generic or right.is_generic)


def _matching_trigger(text: str, phrases: tuple[str, ...]) -> str | None:
    for phrase in phrases:
        if f" {phrase} " in text:
            return phrase
    return None


def _is_diversification_resilience_pair(
    left: EntityMention,
    right: EntityMention,
) -> bool:
    names = {left.canonical_name, right.canonical_name}
    return names == {"Diversification", "Resilience"}
