"""Render a presentation-ready GraphRAG explainer animation as a GIF.

This script builds a clean, deterministic animation that explains the project:
problem -> classical retrieval -> limitations -> graph construction ->
graph-based retrieval -> answer generation -> comparison -> conclusion.

The visuals are grounded in real artifacts already produced in this repo:
- the calibrated answer-level evaluation summary
- the retrieval evaluation summary
- the GraphRAG index statistics
- a real diversification query trace used throughout the animation
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx
from PIL import Image, ImageColor, ImageDraw, ImageFont


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT_DIR / "results" / "animations" / "graphrag_project_explainer.gif"
DEFAULT_POSTER = ROOT_DIR / "results" / "animations" / "graphrag_project_explainer_poster.png"
QUERY_SLUG = "how_does_supply_chain_diversification_improve_resi"
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720


PALETTE = {
    "bg": "#F7F9FC",
    "panel": "#FFFFFF",
    "panel_alt": "#EEF3FB",
    "text": "#162033",
    "muted": "#5F6C80",
    "line": "#D8E0EC",
    "tfidf": "#5A8DEE",
    "dense": "#18A572",
    "graph": "#FF8C42",
    "hybrid": "#8B5CF6",
    "entity": "#4B7BEC",
    "chunk": "#2BAE66",
    "edge": "#AAB6C8",
    "highlight": "#FFB703",
    "danger": "#E85D75",
    "good": "#18A572",
}


@dataclass
class Fonts:
    title: ImageFont.FreeTypeFont | ImageFont.ImageFont
    subtitle: ImageFont.FreeTypeFont | ImageFont.ImageFont
    body: ImageFont.FreeTypeFont | ImageFont.ImageFont
    body_small: ImageFont.FreeTypeFont | ImageFont.ImageFont
    label: ImageFont.FreeTypeFont | ImageFont.ImageFont
    mono: ImageFont.FreeTypeFont | ImageFont.ImageFont


@dataclass
class RunSummary:
    method: str
    chunk_id: str
    title: str
    page_number: int
    preview: str


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_latest(pattern: str) -> Path:
    matches = sorted(ROOT_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find artifact matching: {pattern}")
    return matches[-1]


def find_latest_run(prefix: str, slug: str) -> Path:
    matches = sorted((ROOT_DIR / "results" / "runs").glob(f"{prefix}*{slug}.json"))
    if not matches:
        raise FileNotFoundError(f"Could not find run for prefix={prefix!r}, slug={slug!r}")
    return matches[-1]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        import PIL

        font_dir = Path(PIL.__file__).resolve().parent / "fonts"
        preferred = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        font_path = font_dir / preferred
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    except Exception:
        pass

    mac_fonts = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in mac_fonts:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue

    return ImageFont.load_default()


def build_fonts() -> Fonts:
    return Fonts(
        title=load_font(42, bold=True),
        subtitle=load_font(24, bold=True),
        body=load_font(24),
        body_small=load_font(19),
        label=load_font(18, bold=True),
        mono=load_font(17),
    )


def hex_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    r, g, b = ImageColor.getrgb(color)
    return (r, g, b, alpha)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def ease(t: float) -> float:
    t = clamp(t)
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def stage(progress: float, start: float, end: float) -> float:
    if end <= start:
        return 0.0
    return ease(clamp((progress - start) / (end - start)))


def shorten_text(text: str, max_chars: int) -> str:
    single_line = " ".join(text.split())
    if len(single_line) <= max_chars:
        return single_line
    return single_line[: max_chars - 1].rstrip() + "..."


def count_manifest_docs() -> int:
    manifest_path = ROOT_DIR / "data" / "manifest.jsonl"
    return sum(1 for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())


def safe_metric(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = center[0] - (bbox[2] - bbox[0]) / 2
    y = center[1] - (bbox[3] - bbox[1]) / 2
    draw.text((x, y), text, font=font, fill=fill)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        test = f"{current} {word}"
        if draw.textlength(test, font=font) <= width:
            current = test
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_wrapped_block(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    line_spacing: int = 8,
) -> None:
    x0, y0, x1, y1 = box
    lines = wrap_text(draw, text, font, x1 - x0)
    y = y0
    for line in lines:
        draw.text((x0, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x0, y), line, font=font)
        y = bbox[3] + line_spacing
        if y > y1:
            break


def draw_highlighted_words(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    highlights: dict[str, tuple[int, int, int, int]],
    line_spacing: int = 10,
) -> None:
    x0, y0, x1, y1 = box
    words = text.split()
    lines: list[list[str]] = [[]]
    current_width = 0.0
    max_width = x1 - x0
    for word in words:
        candidate = " ".join(lines[-1] + [word]).strip()
        candidate_width = draw.textlength(candidate, font=font)
        if lines[-1] and candidate_width > max_width:
            lines.append([word])
            current_width = draw.textlength(word, font=font)
        else:
            lines[-1].append(word)
            current_width = candidate_width
    y = y0
    for tokens in lines:
        x = x0
        for token in tokens:
            token_clean = token.strip(".,;:!?()[]{}'\"").lower()
            width = draw.textlength(token, font=font)
            if token_clean in highlights:
                pad_x = 6
                pad_y = 2
                bbox = draw.textbbox((x, y), token, font=font)
                draw.rounded_rectangle(
                    (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y),
                    radius=8,
                    fill=highlights[token_clean],
                )
            draw.text((x, y), token, font=font, fill=fill)
            x += width + draw.textlength(" ", font=font)
        y = draw.textbbox((x0, y), "Ag", font=font)[3] + line_spacing
        if y > y1:
            break


def draw_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str | None = None,
    radius: int = 20,
    alpha: int = 255,
) -> None:
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=hex_rgba(fill, alpha),
        outline=hex_rgba(outline, alpha) if outline else None,
        width=2 if outline else 0,
    )


def draw_badge(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    bg: str,
    fg: str = "#FFFFFF",
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    padding_x = 12
    padding_y = 6
    box = (
        xy[0],
        xy[1],
        xy[0] + (bbox[2] - bbox[0]) + 2 * padding_x,
        xy[1] + (bbox[3] - bbox[1]) + 2 * padding_y,
    )
    draw.rounded_rectangle(box, radius=14, fill=hex_rgba(bg))
    draw.text((xy[0] + padding_x, xy[1] + padding_y - 1), text, font=font, fill=hex_rgba(fg))


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: str,
    width: int = 4,
    alpha: int = 255,
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=hex_rgba(fill, alpha), width=width)
    angle = math.atan2(ey - sy, ex - sx)
    head = 12
    wing = 7
    left = (
        ex - head * math.cos(angle) + wing * math.sin(angle),
        ey - head * math.sin(angle) - wing * math.cos(angle),
    )
    right = (
        ex - head * math.cos(angle) - wing * math.sin(angle),
        ey - head * math.sin(angle) + wing * math.cos(angle),
    )
    draw.polygon([end, left, right], fill=hex_rgba(fill, alpha))


def draw_node(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    radius: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    outline: str = "#FFFFFF",
    text_fill: str = "#FFFFFF",
    alpha: int = 255,
) -> None:
    cx, cy = center
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(box, fill=hex_rgba(fill, alpha), outline=hex_rgba(outline, alpha), width=3)
    draw_centered_text(draw, center, text, font=font, fill=hex_rgba(text_fill, alpha))


def load_project_snapshot() -> dict:
    answer_summary = load_json(find_latest("results/answer_evaluation/answer_eval_*/aggregate_summary.json"))
    retrieval_summary = load_json(find_latest("results/evaluation/retrieval_eval_*/aggregate_summary.json"))
    graph_info = load_json(ROOT_DIR / "results" / "indexes" / "baseline_graph" / "index_info.json")

    tfidf_run = load_json(find_latest_run("baseline_", QUERY_SLUG))
    dense_run = load_json(find_latest_run("baseline_dense_", QUERY_SLUG))
    graph_run = load_json(find_latest_run("baseline_graph_", QUERY_SLUG))
    hybrid_run = load_json(find_latest_run("baseline_graph_hybrid_", QUERY_SLUG))

    summaries = {}
    for method, run in {
        "tfidf": tfidf_run,
        "dense": dense_run,
        "graph": graph_run,
        "hybrid_graph": hybrid_run,
    }.items():
        top = run["retrieved_chunks"][0]
        summaries[method] = RunSummary(
            method=method,
            chunk_id=top["chunk_id"],
            title=top["title"],
            page_number=top["page_number"],
            preview=shorten_text(top["text"], 160),
        )

    method_scores = {
        entry["method_label"]: entry for entry in answer_summary["method_summaries"]
    }
    retrieval_scores = {
        entry["method_label"]: entry for entry in retrieval_summary["method_summaries"]
    }

    category_leaders = {}
    categories = retrieval_summary.get("category_distribution", {}).keys()
    for category in categories:
        best_method = None
        best_score = -1.0
        for method, entry in method_scores.items():
            score = entry["mean_overall_by_category"].get(category, 0.0)
            if score > best_score:
                best_method = method
                best_score = score
        category_leaders[category] = {"method": best_method, "score": best_score}

    return {
        "doc_count": count_manifest_docs(),
        "chunk_count": graph_info["indexed_chunks"],
        "query_text": graph_run["query_text"],
        "graph_info": graph_info,
        "answer_summary": answer_summary,
        "retrieval_summary": retrieval_summary,
        "run_summaries": summaries,
        "graph_run": graph_run,
        "method_scores": method_scores,
        "retrieval_scores": retrieval_scores,
        "category_leaders": category_leaders,
    }


def graph_layout() -> tuple[nx.DiGraph, dict[str, tuple[float, float]]]:
    graph = nx.DiGraph()
    nodes = {
        "Diversification": "strategy",
        "Resilience": "capability",
        "Flexibility": "capability",
        "Disruption": "risk",
        "Information\nSharing": "capability",
        "Visibility": "capability",
        "Trade\nFacilitation": "policy",
        "New\nSuppliers": "system",
        "Services": "sector",
        "UNCTAD\np14": "chunk",
        "OECD\np109": "chunk",
        "WTO\np13": "chunk",
    }
    for node, node_type in nodes.items():
        graph.add_node(node, node_type=node_type)

    edges = [
        ("Diversification", "Resilience", "strengthens"),
        ("Diversification", "Disruption", "mitigates"),
        ("Diversification", "Flexibility", "enables"),
        ("Information\nSharing", "Visibility", "improves"),
        ("Visibility", "Disruption", "detects"),
        ("Trade\nFacilitation", "New\nSuppliers", "helps find"),
        ("Services", "Resilience", "supports"),
        ("Diversification", "UNCTAD\np14", "evidence"),
        ("Resilience", "UNCTAD\np14", "evidence"),
        ("Trade\nFacilitation", "OECD\np109", "evidence"),
        ("New\nSuppliers", "OECD\np109", "evidence"),
        ("Information\nSharing", "WTO\np13", "evidence"),
        ("Visibility", "WTO\np13", "evidence"),
    ]
    for source, target, relation in edges:
        graph.add_edge(source, target, relation=relation)

    pos = nx.spring_layout(graph, seed=42, k=1.7, iterations=200)
    return graph, pos


def normalize_positions(
    positions: dict[str, tuple[float, float]],
    box: tuple[int, int, int, int],
) -> dict[str, tuple[float, float]]:
    x0, y0, x1, y1 = box
    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    normalized = {}
    for node, (x, y) in positions.items():
        nx_ = x0 + 40 + ((x - min_x) / (max_x - min_x + 1e-9)) * (x1 - x0 - 80)
        ny_ = y0 + 40 + ((y - min_y) / (max_y - min_y + 1e-9)) * (y1 - y0 - 80)
        normalized[node] = (nx_, ny_)
    return normalized


def scene_header(
    draw: ImageDraw.ImageDraw,
    fonts: Fonts,
    title: str,
    subtitle: str,
    step: str,
) -> None:
    draw.text((54, 38), title, font=fonts.title, fill=hex_rgba(PALETTE["text"]))
    draw.text((56, 95), subtitle, font=fonts.body_small, fill=hex_rgba(PALETTE["muted"]))
    draw_badge(draw, (1085, 40), step, fonts.label, PALETTE["panel_alt"], PALETTE["text"])


def scene_problem(snapshot: dict, progress: float, size: tuple[int, int], fonts: Fonts) -> Image.Image:
    image = Image.new("RGBA", size, hex_rgba(PALETTE["bg"]))
    draw = ImageDraw.Draw(image)
    scene_header(
        draw,
        fonts,
        "Document Retrieval Is Hard",
        f"{snapshot['doc_count']} public reports became {snapshot['chunk_count']:,} searchable chunks.",
        "1 / 9",
    )

    query_box = (70, 150, 1210, 250)
    draw_card(draw, query_box, PALETTE["panel"], PALETTE["line"])
    draw.text((95, 172), "Question", font=fonts.label, fill=hex_rgba(PALETTE["muted"]))
    draw_wrapped_block(draw, (95, 198, 1170, 238), snapshot["query_text"], fonts.subtitle, hex_rgba(PALETTE["text"]))

    reveal = stage(progress, 0.0, 0.6)
    cols = 5
    rows = 3
    start_x, start_y = 90, 300
    card_w, card_h = 205, 95
    gap_x, gap_y = 24, 24
    total_cards = cols * rows
    active_cards = max(1, int(total_cards * reveal))
    for idx in range(active_cards):
        row = idx // cols
        col = idx % cols
        x = start_x + col * (card_w + gap_x)
        y = start_y + row * (card_h + gap_y)
        fill = PALETTE["panel"] if idx != 7 else "#FFF5E6"
        outline = PALETTE["line"] if idx != 7 else PALETTE["graph"]
        draw_card(draw, (x, y, x + card_w, y + card_h), fill, outline)
        draw.text((x + 16, y + 16), f"Chunk {idx + 1:02d}", font=fonts.label, fill=hex_rgba(PALETTE["text"]))
        preview = "keyword match..." if idx % 3 == 0 else "policy context..." if idx % 3 == 1 else "supply-chain risk..."
        if idx == 7:
            preview = "actual evidence about diversification and resilience"
        draw_wrapped_block(draw, (x + 16, y + 42, x + card_w - 16, y + card_h - 12), preview, fonts.body_small, hex_rgba(PALETTE["muted"]))

    if progress > 0.55:
        draw_arrow(draw, (640, 252), (640, 300), PALETTE["highlight"], width=5)
        draw_badge(draw, (520, 268), "Need the right evidence", fonts.body_small, PALETTE["highlight"], PALETTE["text"])

    caption = "The core problem is not reading documents. It is retrieving the few chunks that directly answer a question."
    draw_wrapped_block(draw, (90, 635, 1180, 690), caption, fonts.body, hex_rgba(PALETTE["text"]))
    return image


def scene_classical(snapshot: dict, progress: float, size: tuple[int, int], fonts: Fonts) -> Image.Image:
    image = Image.new("RGBA", size, hex_rgba(PALETTE["bg"]))
    draw = ImageDraw.Draw(image)
    scene_header(
        draw,
        fonts,
        "Classical Retrieval Baselines",
        "TF-IDF matches words. Dense retrieval matches semantic similarity.",
        "2 / 9",
    )
    draw_badge(draw, (100, 142), "Same question, different signals", fonts.body_small, PALETTE["panel_alt"], PALETTE["text"])

    left = (60, 190, 620, 640)
    right = (660, 190, 1220, 640)
    draw_card(draw, left, PALETTE["panel"], PALETTE["line"])
    draw_card(draw, right, PALETTE["panel"], PALETTE["line"])
    draw.text((88, 214), "TF-IDF", font=fonts.subtitle, fill=hex_rgba(PALETTE["tfidf"]))
    draw.text((690, 214), "Dense Retrieval", font=fonts.subtitle, fill=hex_rgba(PALETTE["dense"]))

    draw_wrapped_block(draw, (88, 252, 590, 286), snapshot["query_text"], fonts.body, hex_rgba(PALETTE["text"]))
    draw_wrapped_block(draw, (690, 252, 1190, 286), snapshot["query_text"], fonts.body, hex_rgba(PALETTE["text"]))

    left_card = (88, 332, 592, 446)
    right_card = (690, 332, 1194, 446)
    draw_card(draw, left_card, "#F8FBFF", PALETTE["tfidf"])
    draw_card(draw, right_card, "#F4FFFB", PALETTE["dense"])
    tfidf = snapshot["run_summaries"]["tfidf"]
    dense = snapshot["run_summaries"]["dense"]
    draw.text((108, 350), tfidf.chunk_id, font=fonts.mono, fill=hex_rgba(PALETTE["tfidf"]))
    draw_wrapped_block(draw, (108, 378, 572, 432), tfidf.preview, fonts.body_small, hex_rgba(PALETTE["text"]))
    draw.text((710, 350), dense.chunk_id, font=fonts.mono, fill=hex_rgba(PALETTE["dense"]))
    draw_wrapped_block(draw, (710, 378, 1174, 432), dense.preview, fonts.body_small, hex_rgba(PALETTE["text"]))

    t1 = stage(progress, 0.15, 0.55)
    t2 = stage(progress, 0.35, 0.8)
    if t1 > 0:
        draw_arrow(draw, (225, 286), (240, 332), PALETTE["tfidf"], width=5, alpha=int(255 * t1))
        draw_arrow(draw, (395, 286), (395, 332), PALETTE["tfidf"], width=5, alpha=int(255 * t1))
        draw_badge(draw, (130, 458), "Exact wording matters", fonts.body_small, PALETTE["tfidf"])
    if t2 > 0:
        draw_arrow(draw, (842, 286), (840, 332), PALETTE["dense"], width=5, alpha=int(255 * t2))
        draw_arrow(draw, (1030, 286), (980, 332), PALETTE["dense"], width=5, alpha=int(255 * t2))
        draw_badge(draw, (760, 458), "Semantic similarity matters", fonts.body_small, PALETTE["dense"])

    draw_wrapped_block(
        draw,
        (90, 565, 1180, 630),
        "Both baselines are useful. But the relationship between concepts still stays implicit inside text.",
        fonts.body,
        hex_rgba(PALETTE["text"]),
    )
    return image


def scene_limitations(snapshot: dict, progress: float, size: tuple[int, int], fonts: Fonts) -> Image.Image:
    image = Image.new("RGBA", size, hex_rgba(PALETTE["bg"]))
    draw = ImageDraw.Draw(image)
    scene_header(
        draw,
        fonts,
        "Where Classical Retrieval Starts To Struggle",
        "Good chunks can still miss the explicit reasoning chain.",
        "3 / 9",
    )

    left = (80, 185, 470, 560)
    mid = (445, 300, 835, 600)
    right = (810, 185, 1200, 560)
    draw_card(draw, left, PALETTE["panel"], PALETTE["line"])
    draw_card(draw, right, PALETTE["panel"], PALETTE["line"])
    draw_card(draw, mid, "#FFF7EC", PALETTE["graph"])
    draw.text((110, 205), "Chunk A", font=fonts.subtitle, fill=hex_rgba(PALETTE["text"]))
    draw.text((840, 205), "Chunk B", font=fonts.subtitle, fill=hex_rgba(PALETTE["text"]))
    draw.text((560, 322), "Missing explicit link", font=fonts.subtitle, fill=hex_rgba(PALETTE["graph"]))

    draw_wrapped_block(
        draw,
        (110, 250, 440, 430),
        "Diversification improves flexibility and helps firms avoid disruption.",
        fonts.body,
        hex_rgba(PALETTE["text"]),
    )
    draw_wrapped_block(
        draw,
        (840, 250, 1170, 430),
        "Resilience is the ability to absorb shocks and recover after disruption.",
        fonts.body,
        hex_rgba(PALETTE["text"]),
    )
    draw_node(draw, (280, 500), 46, "Diversification", fonts.body_small, PALETTE["entity"])
    draw_node(draw, (995, 500), 46, "Resilience", fonts.body_small, PALETTE["entity"])

    dash_progress = stage(progress, 0.15, 0.65)
    for idx in range(int(8 * dash_progress)):
        x = 334 + idx * 75
        draw.line((x, 500, x + 36, 500), fill=hex_rgba(PALETTE["danger"]), width=4)
    if progress > 0.45:
        draw_centered_text(draw, (640, 500), "?", fonts.title, hex_rgba(PALETTE["danger"]))
    draw_wrapped_block(
        draw,
        (528, 370, 760, 520),
        "Keyword and dense retrieval can surface useful text, but they do not make the relation itself a first-class object.",
        fonts.body_small,
        hex_rgba(PALETTE["text"]),
    )
    return image


def scene_graph_construction(snapshot: dict, progress: float, size: tuple[int, int], fonts: Fonts) -> Image.Image:
    image = Image.new("RGBA", size, hex_rgba(PALETTE["bg"]))
    draw = ImageDraw.Draw(image)
    scene_header(
        draw,
        fonts,
        "Build The Knowledge Graph",
        "Start from chunks, extract entities, then add typed relations with evidence.",
        "4 / 9",
    )

    chunk_box = (70, 175, 720, 610)
    draw_card(draw, chunk_box, PALETTE["panel"], PALETTE["line"])
    draw.text((95, 195), "Example chunk from the corpus", font=fonts.subtitle, fill=hex_rgba(PALETTE["text"]))
    graph_chunk = snapshot["run_summaries"]["graph"].preview
    highlights = {
        "diversification": hex_rgba("#FFE5B4"),
        "resilience": hex_rgba("#DBEAFE"),
        "disruption": hex_rgba("#FDE2E1"),
        "flexibility": hex_rgba("#DCFCE7"),
    }
    highlight_strength = stage(progress, 0.0, 0.4)
    active_highlights = {}
    for word, color in highlights.items():
        alpha = int(lerp(0, color[3], highlight_strength))
        active_highlights[word] = (color[0], color[1], color[2], alpha)
    draw_highlighted_words(
        draw,
        (95, 250, 680, 565),
        graph_chunk,
        fonts.body,
        hex_rgba(PALETTE["text"]),
        active_highlights,
    )

    node_stage = stage(progress, 0.28, 0.68)
    edge_stage = stage(progress, 0.55, 0.95)
    if node_stage > 0:
        draw_arrow(draw, (720, 385), (830, 385), PALETTE["graph"], width=5, alpha=int(255 * node_stage))
        draw_badge(draw, (740, 330), "extract entities", fonts.body_small, PALETTE["graph"])
        node_alpha = int(255 * node_stage)
        draw_node(draw, (940, 255), 46, "Diversification", fonts.body_small, PALETTE["entity"], alpha=node_alpha)
        draw_node(draw, (1120, 255), 44, "Resilience", fonts.body_small, PALETTE["entity"], alpha=node_alpha)
        draw_node(draw, (940, 430), 44, "Disruption", fonts.body_small, PALETTE["entity"], alpha=node_alpha)
        draw_node(draw, (1120, 430), 44, "Flexibility", fonts.body_small, PALETTE["entity"], alpha=node_alpha)
    if edge_stage > 0:
        alpha = int(255 * edge_stage)
        draw_arrow(draw, (984, 255), (1074, 255), PALETTE["graph"], width=6, alpha=alpha)
        draw_arrow(draw, (940, 301), (940, 384), PALETTE["graph"], width=6, alpha=alpha)
        draw_arrow(draw, (984, 430), (1074, 430), PALETTE["graph"], width=6, alpha=alpha)
        draw.text((1014, 225), "strengthens", font=fonts.body_small, fill=hex_rgba(PALETTE["graph"], alpha))
        draw.text((952, 340), "mitigates", font=fonts.body_small, fill=hex_rgba(PALETTE["graph"], alpha))
        draw.text((1004, 400), "enables", font=fonts.body_small, fill=hex_rgba(PALETTE["graph"], alpha))
        draw_badge(draw, (870, 520), "edges carry evidence", fonts.body_small, PALETTE["panel_alt"], PALETTE["text"])
    return image


def scene_graph_visual(snapshot: dict, progress: float, size: tuple[int, int], fonts: Fonts) -> Image.Image:
    image = Image.new("RGBA", size, hex_rgba(PALETTE["bg"]))
    draw = ImageDraw.Draw(image)
    scene_header(
        draw,
        fonts,
        "Graph Representation",
        "The project graph stores concepts, chunks, and typed links between them.",
        "5 / 9",
    )
    info = snapshot["graph_info"]
    draw_badge(
        draw,
        (70, 140),
        f"{info['entity_nodes']} entity nodes  |  {info['relation_edges']} relation edges  |  {info['sentence_nodes']} sentence nodes",
        fonts.body_small,
        PALETTE["panel_alt"],
        PALETTE["text"],
    )
    graph, raw_positions = graph_layout()
    positions = normalize_positions(raw_positions, (110, 190, 1170, 630))
    visible_nodes = list(graph.nodes())
    node_reveal = max(1, int(len(visible_nodes) * stage(progress, 0.0, 0.45)))
    edge_reveal = max(0, int(graph.number_of_edges() * stage(progress, 0.28, 0.9)))

    visible_set = set(visible_nodes[:node_reveal])
    for idx, edge in enumerate(graph.edges(data=True)):
        if idx >= edge_reveal:
            break
        source, target, data = edge
        if source not in visible_set or target not in visible_set:
            continue
        color = PALETTE["graph"] if data["relation"] != "evidence" else PALETTE["edge"]
        draw_arrow(draw, positions[source], positions[target], color, width=4, alpha=210)
        mid_x = (positions[source][0] + positions[target][0]) / 2
        mid_y = (positions[source][1] + positions[target][1]) / 2
        draw_badge(draw, (int(mid_x) - 42, int(mid_y) - 14), data["relation"], fonts.body_small, "#FFF4E8", PALETTE["graph"])
    for node in visible_nodes[:node_reveal]:
        node_type = graph.nodes[node]["node_type"]
        if node_type == "chunk":
            fill = PALETTE["chunk"]
            radius = 40
        elif node_type == "risk":
            fill = PALETTE["danger"]
            radius = 38
        elif node_type == "policy":
            fill = PALETTE["hybrid"]
            radius = 42
        else:
            fill = PALETTE["entity"]
            radius = 40
        draw_node(draw, positions[node], radius, node, fonts.body_small, fill)

    draw_wrapped_block(
        draw,
        (100, 648, 1180, 700),
        "This is a simplified view of the real project graph. The actual index is much larger and grounded in extracted chunk evidence.",
        fonts.body_small,
        hex_rgba(PALETTE["muted"]),
    )
    return image


def scene_graph_retrieval(snapshot: dict, progress: float, size: tuple[int, int], fonts: Fonts) -> Image.Image:
    image = Image.new("RGBA", size, hex_rgba(PALETTE["bg"]))
    draw = ImageDraw.Draw(image)
    scene_header(
        draw,
        fonts,
        "GraphRAG Retrieval",
        "Map the query to graph nodes, traverse typed edges, and retrieve supporting chunks.",
        "6 / 9",
    )
    query_box = (70, 160, 1210, 245)
    draw_card(draw, query_box, PALETTE["panel"], PALETTE["line"])
    draw_wrapped_block(draw, (95, 183, 1160, 230), snapshot["query_text"], fonts.subtitle, hex_rgba(PALETTE["text"]))

    query_entities = snapshot["graph_run"]["query_entities"]
    chips_x = 95
    for entity in query_entities:
        label = entity["canonical_name"]
        color = PALETTE["graph"] if label == "Diversification" else PALETTE["entity"]
        draw_badge(draw, (chips_x, 255), label, fonts.body_small, color)
        chips_x += int(draw.textlength(label, font=fonts.body_small)) + 70

    graph, raw_positions = graph_layout()
    positions = normalize_positions(raw_positions, (90, 315, 760, 650))
    selected_path = {
        ("Diversification", "Resilience"),
        ("Diversification", "Disruption"),
        ("Diversification", "UNCTAD\np14"),
        ("Resilience", "UNCTAD\np14"),
    }

    path_stage = stage(progress, 0.18, 0.78)
    for source, target, data in graph.edges(data=True):
        color = PALETTE["highlight"] if (source, target) in selected_path else PALETTE["edge"]
        alpha = 240 if (source, target) in selected_path and path_stage > 0 else 120
        if (source, target) in selected_path:
            alpha = int(255 * path_stage)
        draw_arrow(draw, positions[source], positions[target], color, width=5 if (source, target) in selected_path else 3, alpha=alpha)

    for node in graph.nodes:
        node_type = graph.nodes[node]["node_type"]
        if node_type == "chunk":
            fill = PALETTE["chunk"]
        elif node_type == "risk":
            fill = PALETTE["danger"]
        elif node_type == "policy":
            fill = PALETTE["hybrid"]
        else:
            fill = PALETTE["entity"]
        if node in {"Diversification", "Resilience", "UNCTAD\np14"}:
            fill = PALETTE["highlight"]
        draw_node(draw, positions[node], 38 if node_type != "chunk" else 40, node, fonts.body_small, fill, text_fill="#162033" if fill == PALETTE["highlight"] else "#FFFFFF")

    panel = (820, 300, 1210, 650)
    draw_card(draw, panel, PALETTE["panel"], PALETTE["line"])
    draw.text((845, 322), "Selected evidence", font=fonts.subtitle, fill=hex_rgba(PALETTE["graph"]))
    top_chunk = snapshot["graph_run"]["retrieved_chunks"][0]
    draw.text((845, 366), top_chunk["chunk_id"], font=fonts.mono, fill=hex_rgba(PALETTE["text"]))
    preview = shorten_text(top_chunk["text"], 240)
    draw_wrapped_block(draw, (845, 400, 1180, 520), preview, fonts.body_small, hex_rgba(PALETTE["text"]))
    relation_labels = [rel["relation_type"].lower() for rel in top_chunk["contributing_relations"][:3]]
    draw_wrapped_block(
        draw,
        (845, 545, 1180, 625),
        "Why this chunk scored highly:\n- " + "\n- ".join(relation_labels),
        fonts.body_small,
        hex_rgba(PALETTE["muted"]),
    )
    return image


def scene_answer_generation(snapshot: dict, progress: float, size: tuple[int, int], fonts: Fonts) -> Image.Image:
    image = Image.new("RGBA", size, hex_rgba(PALETTE["bg"]))
    draw = ImageDraw.Draw(image)
    scene_header(
        draw,
        fonts,
        "Grounded Answer Generation",
        "The LLM sees the question plus retrieved evidence and answers with citations.",
        "7 / 9",
    )

    left = (70, 175, 585, 630)
    right = (625, 175, 1210, 630)
    draw_card(draw, left, PALETTE["panel"], PALETTE["line"])
    draw_card(draw, right, PALETTE["panel"], PALETTE["line"])
    draw.text((95, 200), "Retrieved context", font=fonts.subtitle, fill=hex_rgba(PALETTE["text"]))
    draw.text((650, 200), "Generated answer", font=fonts.subtitle, fill=hex_rgba(PALETTE["text"]))

    context_cards = [
        snapshot["run_summaries"]["tfidf"],
        snapshot["run_summaries"]["dense"],
        snapshot["run_summaries"]["graph"],
    ]
    colors = [PALETTE["tfidf"], PALETTE["dense"], PALETTE["graph"]]
    for idx, (item, color) in enumerate(zip(context_cards, colors, strict=False)):
        y = 255 + idx * 110
        box = (95, y, 560, y + 92)
        draw_card(draw, box, PALETTE["panel_alt"], color)
        draw.text((115, y + 15), item.chunk_id, font=fonts.mono, fill=hex_rgba(color))
        draw_wrapped_block(draw, (115, y + 42, 535, y + 82), item.preview, fonts.body_small, hex_rgba(PALETTE["text"]))

    answer = (
        "Supply chain diversification improves resilience by reducing dependence on any one supplier or market, "
        "increasing flexibility, and helping firms mitigate disruptions and absorb shocks "
        "[unctad_gsc_2023:p0014:c002]."
    )
    typed_chars = int(len(answer) * stage(progress, 0.18, 0.88))
    draw_card(draw, (650, 255, 1185, 430), "#F8FBFF", PALETTE["line"])
    draw_wrapped_block(draw, (675, 280, 1150, 410), answer[:typed_chars], fonts.body, hex_rgba(PALETTE["text"]))
    if progress > 0.55:
        draw_badge(draw, (675, 455), "Grounded in retrieved chunks", fonts.body_small, PALETTE["good"])
        draw_badge(draw, (908, 455), "Inspectable citations", fonts.body_small, PALETTE["panel_alt"], PALETTE["text"])
    draw_wrapped_block(
        draw,
        (650, 520, 1170, 610),
        "This step is the same idea for all methods. The retrieval path changes the evidence the model sees.",
        fonts.body_small,
        hex_rgba(PALETTE["muted"]),
    )
    return image


def scene_comparison(snapshot: dict, progress: float, size: tuple[int, int], fonts: Fonts) -> Image.Image:
    image = Image.new("RGBA", size, hex_rgba(PALETTE["bg"]))
    draw = ImageDraw.Draw(image)
    scene_header(
        draw,
        fonts,
        "Method Comparison",
        "Calibrated answer-level evaluation on the 24-question benchmark, top-k = 3.",
        "8 / 9",
    )
    method_scores = snapshot["method_scores"]
    bars = [
        ("TF-IDF", method_scores["tfidf"]["mean_overall_score"], PALETTE["tfidf"]),
        ("Dense", method_scores["dense"]["mean_overall_score"], PALETTE["dense"]),
        ("GraphRAG", method_scores["graph"]["mean_overall_score"], PALETTE["graph"]),
        ("Hybrid GraphRAG", method_scores["hybrid_graph"]["mean_overall_score"], PALETTE["hybrid"]),
    ]
    x = 120
    bar_w = 220
    max_h = 300
    base_y = 580
    reveal = stage(progress, 0.08, 0.7)
    for label, score, color in bars:
        h = int(max_h * score * reveal)
        draw.rounded_rectangle((x, base_y - h, x + bar_w, base_y), radius=18, fill=hex_rgba(color))
        draw.text((x + 18, 598), label, font=fonts.body, fill=hex_rgba(PALETTE["text"]))
        draw.text((x + 18, base_y - h - 36), safe_metric(score), font=fonts.subtitle, fill=hex_rgba(PALETTE["text"]))
        x += 270

    if progress > 0.42:
        draw_badge(draw, (115, 170), "Best overall: TF-IDF", fonts.body_small, PALETTE["tfidf"])
        draw_badge(draw, (390, 170), "Best graph-aware: Hybrid", fonts.body_small, PALETTE["hybrid"])
        draw_badge(draw, (725, 170), "Causal leader: Hybrid", fonts.body_small, PALETTE["graph"])
        draw_badge(draw, (975, 170), "Direct factual leader: TF-IDF", fonts.body_small, PALETTE["panel_alt"], PALETTE["text"])

    draw_wrapped_block(
        draw,
        (110, 635, 1170, 690),
        "In this project, TF-IDF stayed strongest on exact evidence lookup, while Hybrid GraphRAG was the strongest graph-aware system and led on several causal and multi-hop categories.",
        fonts.body,
        hex_rgba(PALETTE["text"]),
    )
    return image


def scene_conclusion(snapshot: dict, progress: float, size: tuple[int, int], fonts: Fonts) -> Image.Image:
    image = Image.new("RGBA", size, hex_rgba(PALETTE["bg"]))
    draw = ImageDraw.Draw(image)
    scene_header(
        draw,
        fonts,
        "Project Takeaway",
        "GraphRAG makes relationships explicit. Hybrid GraphRAG turns that into a stronger final QA system.",
        "9 / 9",
    )

    cards = [
        ("TF-IDF", "Best for exact factual lookup and explicit phrasing.", PALETTE["tfidf"]),
        ("Dense", "Best semantic baseline when wording changes.", PALETTE["dense"]),
        ("Pure GraphRAG", "Best for explaining typed links, but weaker on exact lookups.", PALETTE["graph"]),
        ("Hybrid GraphRAG", "Best final graph-aware method in this repo.", PALETTE["hybrid"]),
    ]
    reveal = stage(progress, 0.0, 0.85)
    for idx, (title, body, color) in enumerate(cards):
        row = idx // 2
        col = idx % 2
        x0 = 90 + col * 560
        y0 = 190 + row * 190
        alpha = int(255 * clamp(reveal * 1.3 - idx * 0.12))
        draw_card(draw, (x0, y0, x0 + 520, y0 + 150), PALETTE["panel"], color, alpha=alpha)
        draw.text((x0 + 26, y0 + 24), title, font=fonts.subtitle, fill=hex_rgba(color))
        draw_wrapped_block(draw, (x0 + 26, y0 + 68, x0 + 490, y0 + 128), body, fonts.body, hex_rgba(PALETTE["text"], alpha))

    if progress > 0.5:
        draw_badge(draw, (385, 590), "Final presentation message: GraphRAG adds explainable structure; hybrid GraphRAG is the best final system.", fonts.body_small, PALETTE["highlight"], PALETTE["text"])
    return image


SCENES = [
    scene_problem,
    scene_classical,
    scene_limitations,
    scene_graph_construction,
    scene_graph_visual,
    scene_graph_retrieval,
    scene_answer_generation,
    scene_comparison,
    scene_conclusion,
]


def render_frames(snapshot: dict, size: tuple[int, int], fonts: Fonts) -> tuple[list[Image.Image], list[int]]:
    frames: list[Image.Image] = []
    durations: list[int] = []
    for scene in SCENES:
        frame_count = 10
        for index in range(frame_count):
            progress = index / (frame_count - 1)
            frames.append(scene(snapshot, progress, size, fonts))
            durations.append(90 if index < frame_count - 1 else 280)
    durations[-1] = 900
    return frames, durations


def save_animation(frames: Sequence[Image.Image], durations: Sequence[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = frames
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=list(durations),
        loop=0,
        optimize=False,
        disposal=2,
    )


def save_poster(frame: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the GraphRAG explainer animation.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output GIF path.")
    parser.add_argument("--poster", type=Path, default=DEFAULT_POSTER, help="Poster PNG path.")
    parser.add_argument("--width", type=int, default=CANVAS_WIDTH, help="Canvas width.")
    parser.add_argument("--height", type=int, default=CANVAS_HEIGHT, help="Canvas height.")
    args = parser.parse_args()

    snapshot = load_project_snapshot()
    fonts = build_fonts()
    frames, durations = render_frames(snapshot, (args.width, args.height), fonts)
    save_animation(frames, durations, args.output)
    save_poster(frames[75], args.poster)

    print(f"Animation written to {args.output}")
    print(f"Poster written to {args.poster}")
    print(f"Frames: {len(frames)}")
    print(f"Canvas: {args.width}x{args.height}")
    print("Scenes:")
    print("- Problem")
    print("- Classical retrieval")
    print("- Limitations")
    print("- Graph construction")
    print("- Graph visualization")
    print("- Graph-based retrieval")
    print("- Answer generation")
    print("- Method comparison")
    print("- Final conclusion")


if __name__ == "__main__":
    main()
