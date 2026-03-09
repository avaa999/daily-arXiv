#!/usr/bin/env python3
"""Generate a daily markdown report for topic/author-focused papers."""

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Set


TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "autonomous exploration": [
        "autonomous exploration",
        "robot exploration",
        "active exploration",
        "frontier exploration",
    ],
    "reinforcement learning": [
        "reinforcement learning",
        "deep reinforcement learning",
        "policy gradient",
        "q-learning",
        "rl",
    ],
    "path planning": [
        "path planning",
        "motion planning",
        "trajectory planning",
        "route planning",
    ],
    "VLM": [
        "vlm",
        "vision-language model",
        "vision language model",
        "vision-language models",
        "multimodal model",
    ],
}

TARGET_AUTHORS: Dict[str, List[str]] = {
    "Gao Fei": ["gao fei", "fei gao"],
    "Zhou Boyu": ["zhou boyu", "boyu zhou"],
    "Cao Yuhong": ["cao yuhong", "yuhong cao"],
    "Daniele Nardi": ["daniele nardi"],
    "Vincenzo Suriani": ["vincenzo suriani"],
    "Guillaume Sartoretti": ["guillaume sartoretti"],
}


def get_bjt_date() -> str:
    """Return date in Asia/Shanghai timezone without external dependency."""
    bjt = timezone(timedelta(hours=8))
    return datetime.now(bjt).strftime("%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily focused arXiv report")
    parser.add_argument("--input", type=str, default="", help="Input jsonl file path")
    parser.add_argument("--date", type=str, default="", help="Report date, e.g. 2026-03-09")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="report",
        help="Directory for output markdown reports",
    )
    return parser.parse_args()


def load_papers(input_file: str) -> List[dict]:
    papers: List[dict] = []
    if not input_file or not os.path.exists(input_file):
        return papers

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                papers.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return papers


def normalize_author_tokens(name: str) -> Set[str]:
    return set(re.findall(r"[a-z]+", name.lower()))


def build_alias_token_map() -> Dict[str, List[Set[str]]]:
    alias_map: Dict[str, List[Set[str]]] = {}
    for target, aliases in TARGET_AUTHORS.items():
        alias_map[target] = [normalize_author_tokens(alias) for alias in aliases]
    return alias_map


def match_target_authors(authors: Iterable[str], alias_token_map: Dict[str, List[Set[str]]]) -> List[str]:
    matched: List[str] = []
    for target, alias_sets in alias_token_map.items():
        for author in authors:
            author_tokens = normalize_author_tokens(str(author))
            if not author_tokens:
                continue
            if any(alias_tokens.issubset(author_tokens) for alias_tokens in alias_sets):
                matched.append(target)
                break
    return matched


def collect_search_text(item: dict) -> str:
    text_fields = [
        item.get("title", ""),
        item.get("summary", ""),
        item.get("comment", ""),
        " ".join(item.get("categories", [])) if isinstance(item.get("categories"), list) else "",
    ]
    ai = item.get("AI", {})
    if isinstance(ai, dict):
        text_fields.extend(str(v) for v in ai.values())
    return "\n".join(str(x) for x in text_fields if x).lower()


def match_topics(text: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in text]
        if hits:
            result[topic] = sorted(set(hits))
    return result


def short_summary(item: dict, limit: int = 220) -> str:
    ai = item.get("AI", {})
    candidate = ""
    if isinstance(ai, dict):
        candidate = str(ai.get("tldr", "")).strip()
    if not candidate:
        candidate = str(item.get("summary", "")).strip().replace("\n", " ")
    if len(candidate) > limit:
        return candidate[: limit - 3] + "..."
    return candidate


def render_report(date_str: str, input_file: str, papers: List[dict]) -> str:
    lines: List[str] = []
    lines.append(f"# arXiv 关键词日报（{date_str}）")
    lines.append("")
    lines.append("> 时区：北京时间（UTC+8）")
    lines.append(
        "> 关注主题关键词：autonomous exploration, reinforcement learning, path planning, VLM"
    )
    lines.append(
        "> 关注作者关键词：Gao Fei, Zhou Boyu, Cao Yuhong, Daniele Nardi, Vincenzo Suriani, Guillaume Sartoretti"
    )
    lines.append("")

    if not papers:
        lines.append("## 今日概览")
        lines.append("")
        if input_file:
            lines.append(f"- 数据文件：`{input_file}` 不存在或为空。")
        else:
            lines.append("- 今日无可用数据文件（可能是去重后无新内容）。")
        lines.append("- 本日报已自动生成，占位说明。")
        return "\n".join(lines) + "\n"

    alias_map = build_alias_token_map()
    matched_records = []
    topic_counter: Counter = Counter()
    author_counter: Counter = Counter()

    for item in papers:
        topics = match_topics(collect_search_text(item))
        authors = item.get("authors", [])
        authors = authors if isinstance(authors, list) else []
        matched_authors = match_target_authors(authors, alias_map)
        if topics or matched_authors:
            matched_records.append(
                {
                    "item": item,
                    "topics": topics,
                    "authors": matched_authors,
                }
            )
            topic_counter.update(topics.keys())
            author_counter.update(matched_authors)

    lines.append("## 今日概览")
    lines.append("")
    lines.append(f"- 扫描论文总数：**{len(papers)}**")
    lines.append(f"- 命中（主题/作者）论文数：**{len(matched_records)}**")
    lines.append("")

    lines.append("### 主题命中统计")
    lines.append("")
    for topic in TOPIC_KEYWORDS:
        lines.append(f"- {topic}: {topic_counter.get(topic, 0)}")
    lines.append("")

    lines.append("### 作者命中统计")
    lines.append("")
    for author in TARGET_AUTHORS:
        lines.append(f"- {author}: {author_counter.get(author, 0)}")
    lines.append("")

    if not matched_records:
        lines.append("## 结论")
        lines.append("")
        lines.append("- 今日未发现与关注主题或作者关键词直接相关的论文。")
        return "\n".join(lines) + "\n"

    lines.append("## 按主题分类")
    lines.append("")
    for topic in TOPIC_KEYWORDS:
        topic_records = [r for r in matched_records if topic in r["topics"]]
        lines.append(f"### {topic}（{len(topic_records)}）")
        lines.append("")
        if not topic_records:
            lines.append("- 无")
            lines.append("")
            continue
        for idx, record in enumerate(topic_records, start=1):
            item = record["item"]
            title = item.get("title", "Untitled")
            abs_url = item.get("abs", "")
            categories = ", ".join(item.get("categories", [])) if isinstance(item.get("categories"), list) else ""
            authors = ", ".join(item.get("authors", [])) if isinstance(item.get("authors"), list) else ""
            matched_topic_terms = ", ".join(record["topics"].get(topic, []))
            matched_author_terms = ", ".join(record["authors"]) if record["authors"] else "无"
            lines.append(f"{idx}. **{title}**")
            if abs_url:
                lines.append(f"   - 链接：{abs_url}")
            lines.append(f"   - 作者：{authors if authors else '未知'}")
            lines.append(f"   - 分类：{categories if categories else '未知'}")
            lines.append(f"   - 主题命中词：{matched_topic_terms}")
            lines.append(f"   - 作者关键词命中：{matched_author_terms}")
            lines.append(f"   - 摘要：{short_summary(item)}")
            lines.append("")

    lines.append("## 按关注作者聚类")
    lines.append("")
    for author in TARGET_AUTHORS:
        author_records = [r for r in matched_records if author in r["authors"]]
        lines.append(f"### {author}（{len(author_records)}）")
        lines.append("")
        if not author_records:
            lines.append("- 无")
            lines.append("")
            continue
        for idx, record in enumerate(author_records, start=1):
            item = record["item"]
            title = item.get("title", "Untitled")
            abs_url = item.get("abs", "")
            topic_tags = ", ".join(record["topics"].keys()) if record["topics"] else "仅作者命中"
            lines.append(f"{idx}. **{title}**")
            if abs_url:
                lines.append(f"   - 链接：{abs_url}")
            lines.append(f"   - 主题关联：{topic_tags}")
            lines.append("")

    lines.append("## 备注")
    lines.append("")
    lines.append("- 命中规则基于关键词字符串匹配与作者名 token 匹配，可能存在少量误报/漏报。")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    date_str = args.date.strip() if args.date else get_bjt_date()
    papers = load_papers(args.input)
    report_text = render_report(date_str=date_str, input_file=args.input, papers=papers)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{date_str}.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Generated report: {output_file}")


if __name__ == "__main__":
    main()
