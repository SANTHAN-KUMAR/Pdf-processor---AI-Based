"""
Enhanced validation script for PDF Structure Extractor

This script provides detailed analysis of extraction results compared to ground truth,
highlighting specific patterns in true/false positives and offering optimization insights.
"""

import json
import os
import glob
import re
from typing import Dict, List, Tuple, Any, Set
from collections import Counter, defaultdict
import difflib


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return {}


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    # Remove excess whitespace and normalize quotes
    text = " ".join(text.split())
    # Replace various quote types with standard quotes
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    return text


def compare_title(extracted_title: str, gold_title: str) -> Dict[str, Any]:
    """
    Compare extracted title with gold standard title.
    
    Returns:
        Dictionary with match result and similarity metrics
    """
    norm_extracted = normalize_text(extracted_title)
    norm_gold = normalize_text(gold_title)
    
    # Calculate string similarity
    similarity = difflib.SequenceMatcher(None, norm_extracted, norm_gold).ratio()
    
    # Check for exact match
    exact_match = norm_extracted == norm_gold
    
    # Check for partial match (>80% similarity or gold is substring of extracted)
    partial_match = similarity > 0.8 or (norm_gold and norm_gold in norm_extracted)
    
    return {
        "exact_match": exact_match,
        "partial_match": partial_match,
        "similarity": similarity,
        "extracted": extracted_title,
        "gold": gold_title
    }


def detailed_heading_comparison(extracted_headings: List[Dict[str, Any]],
                               gold_headings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform detailed comparison of extracted headings with gold standard.
    
    Returns:
        Dictionary with detailed comparison results
    """
    # Normalize headings for comparison
    gold_normalized = [
        {
            "level": h.get("level"),
            "text": normalize_text(h.get("text", "")),
            "page": h.get("page", 0),
            "original": h
        }
        for h in gold_headings
    ]
    
    extracted_normalized = [
        {
            "level": h.get("level"),
            "text": normalize_text(h.get("text", "")),
            "page": h.get("page", 0),
            "original": h
        }
        for h in extracted_headings
    ]
    
    # Track matches
    true_positives = []
    matched_gold_indices = set()
    matched_extracted_indices = set()
    
    # Find exact matches (both text and level)
    for i, ext_heading in enumerate(extracted_normalized):
        for j, gold_heading in enumerate(gold_normalized):
            if j in matched_gold_indices:
                continue
                
            if (ext_heading["level"] == gold_heading["level"] and
                ext_heading["text"] == gold_heading["text"]):
                true_positives.append({
                    "extracted_index": i,
                    "gold_index": j,
                    "heading": ext_heading,
                    "match_type": "exact"
                })
                matched_gold_indices.add(j)
                matched_extracted_indices.add(i)
                break
    
    # Find text-only matches (level might differ)
    for i, ext_heading in enumerate(extracted_normalized):
        if i in matched_extracted_indices:
            continue
            
        for j, gold_heading in enumerate(gold_normalized):
            if j in matched_gold_indices:
                continue
                
            if ext_heading["text"] == gold_heading["text"]:
                true_positives.append({
                    "extracted_index": i,
                    "gold_index": j,
                    "heading": ext_heading,
                    "match_type": "text_only",
                    "expected_level": gold_heading["level"],
                    "actual_level": ext_heading["level"]
                })
                matched_gold_indices.add(j)
                matched_extracted_indices.add(i)
                break
    
    # Collect false positives (extracted but not in gold)
    false_positives = []
    for i, heading in enumerate(extracted_normalized):
        if i not in matched_extracted_indices:
            false_positives.append({
                "index": i,
                "heading": heading
            })
    
    # Collect false negatives (in gold but not extracted)
    false_negatives = []
    for j, heading in enumerate(gold_normalized):
        if j not in matched_gold_indices:
            false_negatives.append({
                "index": j,
                "heading": heading
            })
    
    # Calculate metrics
    precision = len(true_positives) / len(extracted_normalized) if extracted_normalized else 0
    recall = len(true_positives) / len(gold_normalized) if gold_normalized else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def analyze_headings(headings: List[Dict[str, Any]], category: str) -> Dict[str, Any]:
    """
    Analyze characteristics of a set of headings.
    
    Args:
        headings: List of heading objects
        category: Category name (true_positive, false_positive, false_negative)
        
    Returns:
        Dictionary with analysis results
    """
    if not headings:
        # FIX: Ensure a consistent dictionary structure is always returned
        return {"count": 0, "patterns": {}}
        
    # Extract text properties
    text_lengths = [len(h["heading"]["text"]) for h in headings]
    word_counts = [len(h["heading"]["text"].split()) for h in headings]
    
    # Check for patterns
    patterns = {
        "numbered": 0,  # Like "1. Title"
        "decimal": 0,   # Like "1.2 Title"
        "nested": 0,    # Like "1.2.3 Title"
        "all_caps": 0,  # Like "TITLE"
        "capitalized": 0,  # Title Case
        "lowercase_start": 0  # Starts with lowercase
    }
    
    levels = Counter()
    
    for h in headings:
        heading_text = h["heading"]["text"]
        
        # Count heading levels
        if h["heading"].get("level") is not None:
            levels[h["heading"]["level"]] += 1
        
        # Check text patterns
        if re.match(r'^\d+\.\d+\.\d+', heading_text):
            patterns["nested"] += 1
        elif re.match(r'^\d+\.\d+', heading_text):
            patterns["decimal"] += 1
        elif re.match(r'^\d+\.', heading_text):
            patterns["numbered"] += 1
            
        if heading_text.isupper() and len(heading_text) > 3:
            patterns["all_caps"] += 1
            
        if re.match(r'^[A-Z][a-z]', heading_text):
            patterns["capitalized"] += 1
            
        if re.match(r'^[a-z]', heading_text):
            patterns["lowercase_start"] += 1
    
    return {
        "count": len(headings),
        "avg_length": sum(text_lengths) / len(headings) if text_lengths else 0,
        "avg_words": sum(word_counts) / len(word_counts) if word_counts else 0,
        "patterns": patterns,
        "levels": dict(levels)
    }


def analyze_file_results(filename: str, comparison_results: Dict[str, Any], 
                        title_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze results for a single file and generate insights.
    
    Args:
        filename: Name of the file
        comparison_results: Results from heading comparison
        title_results: Results from title comparison
        
    Returns:
        Dictionary with analysis and recommendations
    """
    # Analyze different heading categories
    tp_analysis = analyze_headings(comparison_results["true_positives"], "true_positive")
    fp_analysis = analyze_headings(comparison_results["false_positives"], "false_positive")
    fn_analysis = analyze_headings(comparison_results["false_negatives"], "false_negative")
    
    # Generate insights
    insights = []
    
    # Title insights
    if not title_results["exact_match"]:
        if title_results["partial_match"]:
            insights.append(f"Title was partially detected (similarity: {title_results['similarity']:.2f})")
        else:
            insights.append(f"Title was not detected correctly")
    
    # FIX: Use .get() for safer dictionary access
    fp_count = fp_analysis.get("count", 0)
    fn_count = fn_analysis.get("count", 0)
    tp_count = tp_analysis.get("count", 0)

    # Heading count insights
    if fp_count > fn_count * 2:
        insights.append(f"Too many false positives: {fp_count} detected vs {fn_count} missed")
        
        # Check what's being incorrectly detected
        if fp_analysis.get("patterns", {}).get("lowercase_start", 0) > fp_count * 0.3:
            insights.append("Many false positives start with lowercase (likely regular text)")
            
    elif fn_count > tp_count * 0.5:
        insights.append(f"Significant headings missed: {fn_count} missed vs {tp_count} found")
    
    # Level classification issues
    level_mismatches = [tp for tp in comparison_results["true_positives"] 
                      if tp.get("match_type") == "text_only"]
    if level_mismatches:
        insights.append(f"{len(level_mismatches)} headings matched text but had wrong level")
    
    # Pattern insights
    if fn_analysis.get("patterns", {}).get("all_caps", 0) > fn_count * 0.3:
        insights.append("Missing many ALL CAPS headings")
        
    if fn_analysis.get("patterns", {}).get("decimal", 0) > fn_count * 0.3:
        insights.append("Missing many decimal-numbered headings (like 1.2)")
    
    # Generate recommendations
    recommendations = []
    
    if fp_count > tp_count:
        recommendations.append("Increase heading score threshold to reduce false positives")
        
    if fn_count > 0.2 * (tp_count + fn_count):
        if fn_analysis.get("patterns", {}).get("all_caps", 0) > 0:
            recommendations.append("Improve detection of ALL CAPS headings")
        if fn_analysis.get("patterns", {}).get("decimal", 0) > 0:
            recommendations.append("Improve detection of numbered headings")
            
    if level_mismatches:
        recommendations.append("Refine heading level assignment logic")
    
    return {
        "filename": filename,
        "metrics": {
            "precision": comparison_results["precision"],
            "recall": comparison_results["recall"],
            "f1_score": comparison_results["f1_score"]
        },
        "title": {
            "match": title_results["exact_match"],
            "similarity": title_results["similarity"]
        },
        "heading_analysis": {
            "true_positives": tp_analysis,
            "false_positives": fp_analysis,
            "false_negatives": fn_analysis
        },
        "insights": insights,
        "recommendations": recommendations
    }


def extract_false_positive_examples(comparison_results: Dict[str, Any], limit: int = 5) -> List[str]:
    """Extract example false positives to help with debugging."""
    examples = []
    
    for fp in comparison_results["false_positives"][:limit]:
        heading = fp["heading"]["original"]
        text = heading.get('text', '')
        level = heading.get('level', 'N/A')
        examples.append(f"  • \"{text[:50]}{'...' if len(text) > 50 else ''}\" (level: {level})")
        
    return examples


def extract_false_negative_examples(comparison_results: Dict[str, Any], limit: int = 5) -> List[str]:
    """Extract example false negatives to help with debugging."""
    examples = []
    
    for fn in comparison_results["false_negatives"][:limit]:
        heading = fn["heading"]["original"]
        text = heading.get('text', '')
        level = heading.get('level', 'N/A')
        examples.append(f"  • \"{text[:50]}{'...' if len(text) > 50 else ''}\" (level: {level})")
        
    return examples


def print_file_analysis(analysis: Dict[str, Any], all_comparison_results: Dict[str, Any]) -> None:
    """Print analysis results for a single file."""
    print(f"\n--- File: {analysis['filename']} ---")
    
    # Print title results
    print(f"  Title:")
    match_symbol = "✓" if analysis["title"]["match"] else "✗"
    print(f"    - Match: {match_symbol} (similarity: {analysis['title']['similarity']:.2f})")
    
    # Print heading metrics
    print("  Headings:")
    tp_count = analysis["heading_analysis"]["true_positives"]["count"]
    fp_count = analysis["heading_analysis"]["false_positives"]["count"]
    fn_count = analysis["heading_analysis"]["false_negatives"]["count"]
    
    print(f"    - Ground truth:     {tp_count + fn_count}")
    print(f"    - Extracted:        {tp_count + fp_count}")
    print(f"    - Correctly found:  {tp_count}")
    print(f"    - False positives:  {fp_count}")
    print(f"    - False negatives:  {fn_count}")
    
    # Print metrics
    print("  Metrics:")
    print(f"    - Precision: {analysis['metrics']['precision']:.2%}")
    print(f"    - Recall:    {analysis['metrics']['recall']:.2%}")
    print(f"    - F1-Score:  {analysis['metrics']['f1_score']:.2%}")
    
    # Print insights
    if analysis["insights"]:
        print("  Insights:")
        for insight in analysis["insights"]:
            print(f"    - {insight}")
    
    # Get the specific comparison results for this file
    comparison = all_comparison_results.get(analysis["filename"], {})
    
    # Print false positive examples
    if fp_count > 0 and comparison:
        print("  Example false positives:")
        for example in extract_false_positive_examples(comparison, limit=3):
            print(example)
    
    # Print false negative examples
    if fn_count > 0 and comparison:
        print("  Example missed headings:")
        for example in extract_false_negative_examples(comparison, limit=3):
            print(example)
    
    # Print recommendations
    if analysis["recommendations"]:
        print("  Recommendations:")
        for rec in analysis["recommendations"]:
            print(f"    - {rec}")


def calculate_overall_metrics(file_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall metrics across all files."""
    if not file_analyses:
        return {
            "total_files": 0, "title_accuracy": 0,
            "heading_counts": {"ground_truth": 0, "extracted": 0, "correct": 0},
            "metrics": {"precision": 0, "recall": 0, "f1_score": 0}
        }
        
    total_tp = sum(a["heading_analysis"]["true_positives"]["count"] for a in file_analyses)
    total_fp = sum(a["heading_analysis"]["false_positives"]["count"] for a in file_analyses)
    total_fn = sum(a["heading_analysis"]["false_negatives"]["count"] for a in file_analyses)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    title_matches = sum(1 for a in file_analyses if a["title"]["match"])
    
    return {
        "total_files": len(file_analyses),
        "title_accuracy": title_matches / len(file_analyses) if file_analyses else 0,
        "heading_counts": {
            "ground_truth": total_tp + total_fn,
            "extracted": total_tp + total_fp,
            "correct": total_tp
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    }


def identify_global_patterns(file_analyses: List[Dict[str, Any]]) -> List[str]:
    """Identify global patterns across all files."""
    if not file_analyses:
        return []
    
    patterns = []
    total_fp = sum(a["heading_analysis"]["false_positives"]["count"] for a in file_analyses)
    total_fn = sum(a["heading_analysis"]["false_negatives"]["count"] for a in file_analyses)
    
    if total_fp == 0 and total_fn == 0 and sum(a["heading_analysis"]["true_positives"]["count"] for a in file_analyses) == 0:
        return ["No headings were processed to identify patterns."]
    
    if total_fp > total_fn * 2:
        patterns.append("GLOBAL ISSUE: Model is over-detecting, leading to high false positives.")
        
    fp_lowercase = sum(a["heading_analysis"]["false_positives"].get("patterns", {}).get("lowercase_start", 0) for a in file_analyses)
    if total_fp > 0 and fp_lowercase / total_fp > 0.3:
        patterns.append("PATTERN: Many false positives start with lowercase. Review filtering for regular text.")

    fn_all_caps = sum(a["heading_analysis"]["false_negatives"].get("patterns", {}).get("all_caps", 0) for a in file_analyses)
    if total_fn > 0 and fn_all_caps / total_fn > 0.2:
        patterns.append("PATTERN: Model is commonly missing ALL CAPS headings.")
        
    fn_numbered = sum(a["heading_analysis"]["false_negatives"].get("patterns", {}).get(p, 0) for p in ["numbered", "decimal", "nested"] for a in file_analyses)
    if total_fn > 0 and fn_numbered / total_fn > 0.2:
        patterns.append("PATTERN: Model is commonly missing numbered headings (e.g., '1.2 Section').")
    
    return patterns


def suggest_optimizations(file_analyses: List[Dict[str, Any]], all_comparison_results: Dict[str, Any]) -> List[str]:
    """Suggest specific optimizations based on analysis."""
    if not file_analyses:
        return ["No data to suggest optimizations."]
        
    suggestions = []
    total_fp = sum(a["heading_analysis"]["false_positives"]["count"] for a in file_analyses)
    total_tp = sum(a["heading_analysis"]["true_positives"]["count"] for a in file_analyses)
    
    if total_fp > total_tp:
        suggestions.append("1. High False Positives: Increase heading score threshold to be more strict.")

    fn_patterns = defaultdict(int)
    for a in file_analyses:
        for pattern, count in a["heading_analysis"]["false_negatives"].get("patterns", {}).items():
            fn_patterns[pattern] += count
    
    if fn_patterns["all_caps"] > 0:
        suggestions.append("2. Missing ALL CAPS: Boost score for ALL CAPS text blocks.")
    
    if fn_patterns["numbered"] > 0 or fn_patterns["decimal"] > 0:
        suggestions.append("3. Missing Numbered Headings: Improve regex for numbered patterns.")
    
    level_mismatches = sum(
        1 for a in file_analyses 
        for tp in all_comparison_results.get(a["filename"], {}).get("true_positives", [])
        if tp.get("match_type") == "text_only"
    )
    
    if level_mismatches > 0:
        suggestions.append("4. Level Mismatches: Refine logic for assigning heading levels (e.g., use font size more effectively).")
    
    return suggestions if suggestions else ["Overall performance is strong. Focus on minor edge cases."]


# Main evaluation function
def evaluate_files(extracted_dir: str, gold_dir: str) -> None:
    """
    Evaluate extracted JSON files against gold standard files.
    
    Args:
        extracted_dir: Directory with extracted JSON files
        gold_dir: Directory with gold standard JSON files
    """
    gold_files = glob.glob(os.path.join(gold_dir, "*.json"))
    gold_files.sort()  # Ensure consistent ordering
    
    file_analyses = []
    all_comparison_results = {}
    
    print("\n=== Detailed Evaluation Results ===")
    
    for gold_path in gold_files:
        filename = os.path.basename(gold_path)
        extracted_path = os.path.join(extracted_dir, filename)
        
        if not os.path.exists(extracted_path):
            print(f"Missing extracted file: {filename}")
            continue
            
        gold_data = load_json_file(gold_path)
        extracted_data = load_json_file(extracted_path)
        
        if not gold_data or not extracted_data:
            continue
        
        title_results = compare_title(extracted_data.get("title", ""), gold_data.get("title", ""))
        
        heading_comparison = detailed_heading_comparison(
            extracted_data.get("outline", []),
            gold_data.get("outline", [])
        )
        
        all_comparison_results[filename] = heading_comparison
        
        file_analysis = analyze_file_results(filename, heading_comparison, title_results)
        file_analyses.append(file_analysis)
        
        print_file_analysis(file_analysis, all_comparison_results)
    
    overall_metrics = calculate_overall_metrics(file_analyses)
    global_patterns = identify_global_patterns(file_analyses)
    optimization_suggestions = suggest_optimizations(file_analyses, all_comparison_results)
    
    print("\n\n==========================")
    print("=== EVALUATION SUMMARY ===")
    print("==========================")
    print(f"\nTotal Files Evaluated: {overall_metrics['total_files']}")
    print(f"Title Accuracy: {overall_metrics['title_accuracy']:.2%}")
    print("\n--- Outline Heading Performance ---")
    print(f"  - Precision: {overall_metrics['metrics']['precision']:.2%}")
    print(f"  - Recall:    {overall_metrics['metrics']['recall']:.2%}")
    print(f"  - F1-Score:  {overall_metrics['metrics']['f1_score']:.2%}")
    print(f"  ({overall_metrics['heading_counts']['correct']} correct out of {overall_metrics['heading_counts']['ground_truth']} ground truth headings)")

    if global_patterns:
        print("\n--- Global Patterns & Issues ---")
        for pattern in global_patterns:
            print(f"  • {pattern}")
    
    print("\n--- Suggested Optimizations ---")
    for suggestion in optimization_suggestions:
        print(f"  • {suggestion}")
    print("\n==========================")


if __name__ == "__main__":
    # Define directories
    EXTRACTED_DIR = "/app/output"
    GOLD_DIR = "/app/ground_truths"
    
    # For local testing:
    if not os.path.exists(EXTRACTED_DIR):
        EXTRACTED_DIR = "./output"
        GOLD_DIR = "./ground_truths"
    
    print("Validating extracted PDF structure against gold standard...")
    evaluate_files(EXTRACTED_DIR, GOLD_DIR)
