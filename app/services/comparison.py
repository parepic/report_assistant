from __future__ import annotations

from typing import List, Dict, Tuple, Any
import os
from pathlib import Path
import json

import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sqlalchemy import select

from app.data_classes import GlobalConfig, compute_strategy_hash
from app.clients.QdrantClientWrapper import QdrantClientWrapper
from app.clients.OpenAiClientWrapper import OpenAIClientWrapper
from app.models import Document
from sqlalchemy.ext.asyncio import AsyncSession


def _build_single_prompt(
    change_type: str,
    risk_factor_title: str,
    paragraph_text: str,
) -> str:
    return f"""
You are a “10-K Risk Factor Delta Summarizer (analyst-grade)”.

Goal: extract ONLY the most decision-relevant, explicitly stated risk MECHANISMS.
Do NOT paraphrase the paragraph. Do NOT restate background or examples unless they add a new obligation/cost/trigger.

------------------------
INPUT
------------------------
Change type: {change_type}   # ADDED | REMOVED
Risk factor title: {risk_factor_title}
Paragraph:
{paragraph_text}

------------------------
OUTPUT (STRICT)
------------------------
Format:
- Bullet points only. No headings, no intro, no prose.
- Prefix each bullet:
  [+] if Change type = ADDED
  [-] if Change type = REMOVED

Selection rules (this is the main task):
- Output only the TOP 2–5 distinct mechanisms by materiality.
- A “mechanism” must be one of:
  (a) a trigger/condition → outcome
  (b) a constraint/obligation (covenant, approval, fee, financing, timetable dependency)
  (c) a named risk driver (integration difficulty, divestiture requirement, enforcement action)
  (d) a concrete business/financial impact (cost increase, reduced flexibility, delayed closing)

Compression rules:
- NO “mirror bullets” / NO reworded duplicates.
- Collapse lists into one bullet when they support the same mechanism
  (e.g., multiple facilities → “committed financing facilities totaling ~$X”).
- Prefer ONE number only when it is a key term (termination fee, total commitment, covenant/limit).
- Do NOT include generic filler like “may adversely affect results” unless tied to a specific trigger.

Wording rules:
- ≤14 words per bullet.
- Use text-close language; keep it concrete.

If the paragraph contains no explicit mechanism meeting the rules, output exactly:
No specific risk mechanism stated.
""".strip()


def _build_pair_prompt(new_text: str, old_text: str) -> str:
    return f"""
You are a Material Risk Change Detector for 10-K risk factor paragraphs.

Task: Compare OLD vs NEW and report ONLY changes that alter the risk meaning for investors.
Do NOT list “more detail” or paraphrases unless they change exposure, magnitude, scope, or outlook.

Count as MATERIAL only if the change introduces/removes/changes:
• Regulatory/legal exposure: new/removed obligations, enforcement posture, investigations, or explicit fines/penalties/settlements.
• Scope/applicability: the risk now applies to different activities, products/services, regions, customers/users, or processes.
• Magnitude: any new/removed/changed quantitative figure, threshold, or concrete quantified example.
• Outlook/timing: changes in years/time horizon or likelihood language (“expect/will” vs “may/could”), persistence, trend.
• New risk driver or constraint: a genuinely new cause of harm, dependency, or negative pathway (not just elaboration).

NOT material (ignore completely):
• Rewording, synonyms, reordering, clarity, added background
• Extra examples that do not add a new risk driver, obligation, scope, magnitude, or outlook
• Minor qualifiers that don’t change likelihood/severity

Output:
- Bullet points only
- Tags: [+] added, [-] removed, [~] changed
- ≤18 words each, delta-only, concrete terms (numbers/years/obligations/scope)
- Max 8 bullets
- If nothing material: output exactly “No material risk change.”

NEW:
{new_text}

OLD:
{old_text}

Return the material risk change bullets.
""".strip()


def _add_llm_outputs(change_result: Dict[str, List[Dict[str, Any]]], config: GlobalConfig, openai_client: OpenAIClientWrapper) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        for bucket in ("changed", "added", "removed"):
            for item in change_result.get(bucket, []):
                item["llm_output"] = "Missing OPENAI_API_KEY"
        return

    for item in change_result.get("added", []):
        payload = item.get("added", {})
        prompt = _build_single_prompt(
            "ADDED",
            payload.get("risk_factor") or "",
            payload.get("text") or "",
        )
        try:
            item["llm_output"] = openai_client.llm_generate(prompt)
        except Exception as exc:
            item["llm_output"] = f"LLM error: {exc}"

    for item in change_result.get("removed", []):
        payload = item.get("removed", {})
        prompt = _build_single_prompt(
            "REMOVED",
            payload.get("risk_factor") or "",
            payload.get("text") or "",
        )
        try:
            item["llm_output"] =  openai_client.llm_generate(prompt)
        except Exception as exc:
            item["llm_output"] = f"LLM error: {exc}"

    for item in change_result.get("changed", []):
        new_payload = item.get("new", {})
        old_payload = item.get("removed") or item.get("old", {})
        prompt = _build_pair_prompt(
            new_payload.get("text") or "",
            old_payload.get("text") or "",
        )
        try:
            item["llm_output"] =  openai_client.llm_generate(prompt)
        except Exception as exc:
            item["llm_output"] = f"LLM error: {exc}"


def get_paragraphs_from_qdrant(
    qdrant_client: QdrantClientWrapper,
    collection_name: str,
    doc_id: str
) -> List[Dict[str, Any]]:
    """
    Retrieve all paragraphs for a given document from Qdrant.

    Returns list of dicts with: id, text, chunk_idx, etc.
    """
    scroll_filter = Filter(must=[
        FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
    ])

    paragraphs = []
    offset = None
    limit = 100

    while True:
        results, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_vectors=False,
            with_payload=True
        )

        if not results:
            break

        for point in results:
            paragraphs.append({
                "id": point.id,
                "text": point.payload.get("text", ""),
                "chunk_idx": point.payload.get("chunk_idx", 0),
                "doc_id": point.payload.get("doc_id", ""),
                "risk_factor": point.payload.get("risk_factor"),
            })

        offset = next_offset
        if offset is None:
            break

    return sorted(paragraphs, key=lambda p: p["chunk_idx"])


def _normalize_words(text: str) -> List[str]:
    cleaned = "".join([c.lower() if c.isalpha() or c == " " else " " for c in text])
    cleaned = " ".join(cleaned.split())
    return cleaned.split(" ") if cleaned else []


def lcws_length(text1: str, text2: str) -> int:
    """Longest common word subsequence length."""
    words1 = _normalize_words(text1)
    words2 = _normalize_words(text2)

    if not words1 or not words2:
        return 0

    dp = np.zeros((len(words1) + 1, len(words2) + 1), dtype=int)
    for i in range(1, len(words1) + 1):
        w1 = words1[i - 1]
        for j in range(1, len(words2) + 1):
            if w1 == words2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return int(dp[len(words1), len(words2)])


def lcws_score(text1: str, text2: str) -> float:
    """Normalized LCWS score in [0, 1], divided by min length."""
    words1 = _normalize_words(text1)
    words2 = _normalize_words(text2)
    if not words1 or not words2:
        return 0.0
    lcs = lcws_length(text1, text2)
    return lcs / min(len(words1), len(words2))


def match_paragraphs_best_one_way(
    qdrant_client: QdrantClientWrapper,
    collection_name: str,
    paras_src: List[Dict[str, Any]],
    paras_tgt: List[Dict[str, Any]],
    doc_id_tgt: str,
    top_k: int,
    strategy_hash: str
) -> Dict[int, Tuple[int, float]]:
    """
    Match each paragraph in source to exactly ONE best paragraph in target,
    but only evaluate LCWS against top-k similar candidates from Qdrant.

    Returns:
        Dict mapping src_idx -> (tgt_idx, score)
    """
    matches: Dict[int, Tuple[int, float]] = {}
    tgt_id_to_idx = {p["id"]: idx for idx, p in enumerate(paras_tgt)}

    for src_idx, para_src in enumerate(paras_src):
        points = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[para_src["id"]],
            with_vectors=True,
            with_payload=False
        )
        if not points:
            continue

        vector = points[0].vector
        if isinstance(vector, dict):
            vector = next(iter(vector.values()), None)
        if vector is None:
            continue
        candidates = qdrant_client.fetch_top_k_vector(
            collection_name=collection_name,
            vector=vector,
            doc_id=doc_id_tgt,
            k=top_k,
            strategy_hash=strategy_hash
        )
        best_match_idx = None
        best_score = 0.0
        for cand in candidates:
            cand_idx = tgt_id_to_idx.get(cand.get("id"))
            if cand_idx is None:
                continue
            score = lcws_score(para_src["text"], paras_tgt[cand_idx]["text"])
            if score > best_score:
                best_score = score
                best_match_idx = cand_idx

        if best_match_idx is not None:
            matches[src_idx] = (best_match_idx, best_score)

    return matches


def find_merged_old_paragraphs(
    old_matches: Dict[int, Tuple[int, float]],
    paras_old: List[Dict[str, Any]]
) -> List[Dict]:
    """
    Find old paragraphs that:
    1. Point to the same new paragraph
    2. Are in the same risk_factor
    
    Returns:
        List of merge groups, each with merged old indices
    """
    # Group by (new_ref_idx, risk_factor)
    groups = {}
    
    for old_idx, (new_ref_idx, score) in old_matches.items():
        risk_factor = paras_old[old_idx].get('risk_factor')
        key = (new_ref_idx, risk_factor)
        
        if key not in groups:
            groups[key] = []
        groups[key].append((old_idx, score))
    
    # Find groups with 2+ paragraphs
    merged_groups = []
    for (new_ref_idx, risk_factor), indices in groups.items():
        if len(indices) >= 2:
            old_indices = [old_idx for old_idx, _ in indices]
            print(
                "Merged group found:",
                f"new_idx={new_ref_idx}",
                f"risk_factor={risk_factor}",
                f"old_indices={old_indices}",
            )
            for old_idx in old_indices:
                para_text = paras_old[old_idx].get("text", "")
                print(f"- old[{old_idx}]: {para_text}")
            merged_groups.append({
                "points_to_new_idx": new_ref_idx,
                "risk_factor": risk_factor,
                "merged_old_indices": indices
            })
    
    return merged_groups



def _update_matches_for_merged_groups(
    new_to_old_matches: Dict[int, Tuple[int, float]],
    paras_new: List[Dict[str, Any]],
    paras_old: List[Dict[str, Any]],
    merged_groups: List[Dict]
) -> Dict[int, Tuple]:
    """
    Update new→old matches when old paragraphs merged.
    For each new para that pointed to a merged old group:
    - Combine texts of all merged old paras
    - Recalculate score with merged text
    - Return match as (merged_group_dict, score) instead of (old_idx, score)
    
    Returns:
        Dict mapping new_idx → (match_info, score)
        where match_info is either old_idx (int) or merged_group_dict (dict)
    """
    # Build reverse map: which new paras pointed to which old paras in each group
    merged_new_paras = {}  # maps new_idx -> merged_group_dict
    
    for merged_group in merged_groups:
        new_ref_idx = merged_group["points_to_new_idx"]
        merged_new_paras[new_ref_idx] = merged_group
    
    # Update matches
    updated = {}
    for new_idx, (old_idx, old_score) in new_to_old_matches.items():
        if new_idx in merged_new_paras:
            # This new para pointed to an old para that's part of a merged group
            merged_group = merged_new_paras[new_idx]
            old_indices_in_group = merged_group["merged_old_indices"]
            
            # Combine texts of all old paras in the merged group
            merged_text = " ".join([
                paras_old[old_i]["text"] 
                for old_i, _ in old_indices_in_group
            ])
            
            # Recalculate score between new and merged text
            new_text = paras_new[new_idx]["text"]
            merged_score = lcws_score(new_text, merged_text)
            
            # Store with merged group info
            updated[new_idx] = (merged_group, merged_score)
        else:
            # No merge for this one
            updated[new_idx] = (old_idx, old_score)
    
    return updated



def _build_old_payload(
    match_info: Any,
    paras_old: List[Dict[str, Any]]
) -> Dict[str, Any]:
    if isinstance(match_info, dict):
        merged_indices = match_info.get("merged_old_indices", [])
        if not merged_indices:
            return {"text": "", "risk_factor": None}

        merged_text = " ".join(
            [paras_old[old_idx]["text"] for old_idx, _ in merged_indices]
        )
        first_old = paras_old[merged_indices[0][0]]
        return {
            "text": merged_text,
            "risk_factor": match_info.get("risk_factor") or first_old.get("risk_factor"),
        }

    old_idx = match_info
    para_old = paras_old[old_idx]
    return {
        "text": para_old.get("text", ""),
        "risk_factor": para_old.get("risk_factor"),
    }


def compare_documents(
    config: GlobalConfig,
    doc_id_old: str,
    doc_id_new: str,
    top_k: int = 10,
    strategy_hash: str = None,
    qdrant_client: QdrantClientWrapper = None,
    openai_client: OpenAIClientWrapper = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compare two documents:
    1. Match each new para to best old para (LCWS over top-k Qdrant candidates)
    2. Match each old para to best new para (LCWS over top-k Qdrant candidates)
    3. Find old paragraphs that point to same new para AND are in same risk_factor
    4. For merged old groups, recalculate similarity with merged text
    5. Save changed, added, and removed paragraphs
    
    Writes results to text files.
    """
    collection_name = config.QDRANT_DB_NAME_YOY or "report_assistant_yoy"

    paras_old = get_paragraphs_from_qdrant(qdrant_client, collection_name, doc_id_old)
    paras_new = get_paragraphs_from_qdrant(qdrant_client, collection_name, doc_id_new)

    print(f"Old paragraphs: {len(paras_old)}")
    print(f"New paragraphs: {len(paras_new)}")
    
    # Match new -> old (each new para gets best old match)
    print("\nMatching new → old...")
    new_to_old_matches = match_paragraphs_best_one_way(
        qdrant_client,
        collection_name,
        paras_new,
        paras_old,
        doc_id_old,
        top_k,
        strategy_hash
    )
    print(f"New paragraphs with matches: {len(new_to_old_matches)}")
    
    # Match old -> new (each old para gets best new match)
    print("Matching old → new...")
    old_to_new_matches = match_paragraphs_best_one_way(
        qdrant_client,
        collection_name,
        paras_old,
        paras_new,
        doc_id_new,
        top_k,
        strategy_hash
    )
    print(f"Old paragraphs with matches: {len(old_to_new_matches)}")
    
    # Find merged old paragraphs
    print("\nIdentifying merged old paragraphs...")
    merged_old_groups = find_merged_old_paragraphs(old_to_new_matches, paras_old)
    print(f"Merged groups found: {len(merged_old_groups)}")
    
    # Update matches for merged old groups: recalculate scores
    print("Updating matches for merged old paragraphs...")
    updated_matches = _update_matches_for_merged_groups(
        new_to_old_matches, paras_new, paras_old, merged_old_groups
    )
    print(f"Updated matches: {len(updated_matches)}")

    change_result: Dict[str, List[Dict[str, Any]]] = {
        "changed": [],
        "added": [],
        "removed": [],
    }

    for new_idx, (match_info, score) in updated_matches.items():
        if score < 0.6:
            bucket = "added"
        elif 0.6 <= score <= 0.9:
            bucket = "changed"
        else:
            continue

        para_new = paras_new[new_idx]
        payload_new = {
            "text": para_new.get("text", ""),
            "risk_factor": para_new.get("risk_factor"),
        }

        if bucket == "changed":
            change_result[bucket].append(
                {
                    "new": payload_new,
                    "removed": _build_old_payload(match_info, paras_old),
                    "similarity": score,
                }
            )
        else:
            change_result[bucket].append(
                {
                    "added": payload_new,
                    "similarity": score,
                }
            )

    for old_idx, (new_idx, score) in old_to_new_matches.items():
        if score >= 0.6:
            continue
        para_old = paras_old[old_idx]
        change_result["removed"].append(
            {
                "removed": {
                    "text": para_old.get("text", ""),
                    "risk_factor": para_old.get("risk_factor"),
                },
                "similarity": score,
            }
        )
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    change_report_file = output_dir / f"yoy_change_{doc_id_old}_vs_{doc_id_new}.txt"
    change_report_file.write_text(
        json.dumps(change_result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    _add_llm_outputs(change_result, config, openai_client)
    
    return change_result



async def main(
    config: GlobalConfig,
    doc_id: str,
    qdrant_client: QdrantClientWrapper,
    openai_client: OpenAIClientWrapper,
    session: AsyncSession
) -> Dict[str, Any]:

    doc_new = await session.get(Document, doc_id)
    if not doc_new:
        raise ValueError(f"Document with id '{doc_id}' not found in database")

    company = doc_new.company
    prev_year = int(doc_new.fiscal_year) - 1

    result = await session.execute(
        select(Document).where(
            Document.company == company,
            Document.fiscal_year == prev_year,
        )
    )
    doc_old = result.scalar_one_or_none()
    if not doc_old:
        raise ValueError(
            f"Document not found for company '{company}' and year '{prev_year}'"
        )

    doc_id_old = doc_old.id

    print("LCWS paragraph matching")
    print(f"Comparing: {doc_id} vs {doc_id_old}")
    strategy_hash = compute_strategy_hash(config.chunk_strategy_yoy)
    return compare_documents(
        config,
        doc_id_old=doc_id_old,
        doc_id_new=doc_id,
        strategy_hash=strategy_hash,
        qdrant_client=qdrant_client,
        openai_client=openai_client,
    )
