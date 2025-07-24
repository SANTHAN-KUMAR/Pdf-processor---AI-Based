#!/usr/bin/env python3
"""
Output validation script for PDF outline extraction.

This script compares generated JSON outputs against ground truth files
and provides detailed accuracy metrics.
"""

import os
import json
import argparse
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import difflib


class OutlineValidator:
    """Validates extracted outlines against ground truth."""
    
    def __init__(self):
        self.metrics = {
            'title_accuracy': 0.0,
            'heading_precision': 0.0,
            'heading_recall': 0.0,
            'heading_f1': 0.0,
            'level_accuracy': 0.0,
            'page_accuracy': 0.0,
            'exact_matches': 0,
            'total_files': 0
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        
        # Convert to lowercase and strip whitespace
        normalized = text.lower().strip()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove common punctuation variations
        for char in '.,;:!?':
            normalized = normalized.replace(char, '')
        
        return normalized
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using sequence matching."""
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        if not norm1 and not norm2:
            return 1.0
        
        if not norm1 or not norm2:
            return 0.0
        
        # Use sequence matcher for similarity
        matcher = difflib.SequenceMatcher(None, norm1, norm2)
        return matcher.ratio()
    
    def match_headings(self, predicted: List[Dict], ground_truth: List[Dict], 
                      similarity_threshold: float = 0.8) -> Tuple[List[Tuple], List[int], List[int]]:
        """Match predicted headings to ground truth based on text similarity."""
        matches = []
        matched_pred = set()
        matched_gt = set()
        
        # Try to match each predicted heading to ground truth
        for i, pred_heading in enumerate(predicted):
            best_match = None
            best_similarity = 0.0
            
            for j, gt_heading in enumerate(ground_truth):
                if j in matched_gt:
                    continue
                
                similarity = self.calculate_text_similarity(
                    pred_heading.get('text', ''),
                    gt_heading.get('text', '')
                )
                
                if similarity > best_similarity and similarity >= similarity_threshold:
                    best_similarity = similarity
                    best_match = j
            
            if best_match is not None:
                matches.append((i, best_match, best_similarity))
                matched_pred.add(i)
                matched_gt.add(best_match)
        
        # Unmatched predictions and ground truth
        unmatched_pred = [i for i in range(len(predicted)) if i not in matched_pred]
        unmatched_gt = [i for i in range(len(ground_truth)) if i not in matched_gt]
        
        return matches, unmatched_pred, unmatched_gt
    
    def evaluate_single_file(self, predicted_path: str, ground_truth_path: str) -> Dict[str, Any]:
        """Evaluate a single predicted output against ground truth."""
        try:
            # Load files
            with open(predicted_path, 'r', encoding='utf-8') as f:
                predicted = json.load(f)
            
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            
        except Exception as e:
            return {
                'error': f"Failed to load files: {e}",
                'title_match': False,
                'heading_metrics': {'precision': 0, 'recall': 0, 'f1': 0},
                'level_accuracy': 0,
                'page_accuracy': 0
            }
        
        results = {}
        
        # Evaluate title
        pred_title = predicted.get('title', '')
        gt_title = ground_truth.get('title', '')
        title_similarity = self.calculate_text_similarity(pred_title, gt_title)
        results['title_match'] = title_similarity >= 0.8
        results['title_similarity'] = title_similarity
        
        # Evaluate headings
        pred_outline = predicted.get('outline', [])
        gt_outline = ground_truth.get('outline', [])
        
        if not gt_outline:
            # No ground truth headings
            results['heading_metrics'] = {
                'precision': 1.0 if not pred_outline else 0.0,
                'recall': 1.0,
                'f1': 1.0 if not pred_outline else 0.0
            }
            results['level_accuracy'] = 1.0
            results['page_accuracy'] = 1.0
        else:
            # Match headings
            matches, unmatched_pred, unmatched_gt = self.match_headings(pred_outline, gt_outline)
            
            # Calculate precision, recall, F1
            true_positives = len(matches)
            false_positives = len(unmatched_pred)
            false_negatives = len(unmatched_gt)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results['heading_metrics'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            }
            
            # Evaluate level and page accuracy for matched headings
            level_correct = 0
            page_correct = 0
            
            for pred_idx, gt_idx, similarity in matches:
                pred_heading = pred_outline[pred_idx]
                gt_heading = gt_outline[gt_idx]
                
                if pred_heading.get('level') == gt_heading.get('level'):
                    level_correct += 1
                
                if pred_heading.get('page') == gt_heading.get('page'):
                    page_correct += 1
            
            results['level_accuracy'] = level_correct / len(matches) if matches else 0
            results['page_accuracy'] = page_correct / len(matches) if matches else 0
        
        # Detailed comparison for debugging
        results['detailed_comparison'] = {
            'predicted_title': pred_title,
            'ground_truth_title': gt_title,
            'predicted_headings': len(pred_outline),
            'ground_truth_headings': len(gt_outline),
            'matches': len(matches) if 'matches' in locals() else 0
        }
        
        return results
    
    def evaluate_directory(self, predicted_dir: str, ground_truth_dir: str) -> Dict[str, Any]:
        """Evaluate all files in directories."""
        print(f"🔍 Validating outputs...")
        print(f"📂 Predicted: {predicted_dir}")
        print(f"📂 Ground truth: {ground_truth_dir}")
        
        # Find matching files
        pred_files = {f.replace('.json', ''): f for f in os.listdir(predicted_dir) if f.endswith('.json')}
        gt_files = {f.replace('.json', ''): f for f in os.listdir(ground_truth_dir) if f.endswith('.json')}
        
        common_files = set(pred_files.keys()) & set(gt_files.keys())
        
        if not common_files:
            print("❌ No matching files found")
            return {'error': 'No matching files found'}
        
        print(f"📄 Found {len(common_files)} matching files to validate")
        
        # Evaluate each file
        file_results = {}
        aggregate_metrics = {
            'title_matches': 0,
            'total_precision': 0,
            'total_recall': 0,
            'total_f1': 0,
            'total_level_acc': 0,
            'total_page_acc': 0,
            'files_processed': 0
        }
        
        for file_key in sorted(common_files):
            pred_path = os.path.join(predicted_dir, pred_files[file_key])
            gt_path = os.path.join(ground_truth_dir, gt_files[file_key])
            
            print(f"📄 Validating {file_key}...")
            
            result = self.evaluate_single_file(pred_path, gt_path)
            file_results[file_key] = result
            
            if 'error' not in result:
                # Aggregate metrics
                if result['title_match']:
                    aggregate_metrics['title_matches'] += 1
                
                aggregate_metrics['total_precision'] += result['heading_metrics']['precision']
                aggregate_metrics['total_recall'] += result['heading_metrics']['recall']
                aggregate_metrics['total_f1'] += result['heading_metrics']['f1']
                aggregate_metrics['total_level_acc'] += result['level_accuracy']
                aggregate_metrics['total_page_acc'] += result['page_accuracy']
                aggregate_metrics['files_processed'] += 1
            
            # Print quick summary
            if 'error' in result:
                print(f"  ❌ Error: {result['error']}")
            else:
                print(f"  📊 Title: {'✅' if result['title_match'] else '❌'} "
                      f"F1: {result['heading_metrics']['f1']:.3f} "
                      f"Level: {result['level_accuracy']:.3f}")
        
        # Calculate final metrics
        if aggregate_metrics['files_processed'] > 0:
            final_metrics = {
                'title_accuracy': aggregate_metrics['title_matches'] / aggregate_metrics['files_processed'],
                'avg_precision': aggregate_metrics['total_precision'] / aggregate_metrics['files_processed'],
                'avg_recall': aggregate_metrics['total_recall'] / aggregate_metrics['files_processed'],
                'avg_f1': aggregate_metrics['total_f1'] / aggregate_metrics['files_processed'],
                'avg_level_accuracy': aggregate_metrics['total_level_acc'] / aggregate_metrics['files_processed'],
                'avg_page_accuracy': aggregate_metrics['total_page_acc'] / aggregate_metrics['files_processed'],
                'files_processed': aggregate_metrics['files_processed'],
                'total_files': len(common_files)
            }
        else:
            final_metrics = {
                'title_accuracy': 0,
                'avg_precision': 0,
                'avg_recall': 0,
                'avg_f1': 0,
                'avg_level_accuracy': 0,
                'avg_page_accuracy': 0,
                'files_processed': 0,
                'total_files': len(common_files)
            }
        
        return {
            'summary': final_metrics,
            'file_results': file_results
        }
    
    def print_summary_report(self, results: Dict[str, Any]):
        """Print detailed summary report."""
        if 'error' in results:
            print(f"❌ Validation failed: {results['error']}")
            return
        
        summary = results['summary']
        
        print("\n" + "="*60)
        print("📊 VALIDATION SUMMARY")
        print("="*60)
        
        print(f"📄 Files processed: {summary['files_processed']}/{summary['total_files']}")
        print(f"📝 Title accuracy: {summary['title_accuracy']:.1%}")
        print(f"🎯 Heading precision: {summary['avg_precision']:.1%}")
        print(f"🔍 Heading recall: {summary['avg_recall']:.1%}")
        print(f"⚖️  Heading F1-score: {summary['avg_f1']:.1%}")
        print(f"📐 Level accuracy: {summary['avg_level_accuracy']:.1%}")
        print(f"📄 Page accuracy: {summary['avg_page_accuracy']:.1%}")
        
        # Performance assessment
        print("\n" + "-"*40)
        print("🎯 PERFORMANCE ASSESSMENT")
        print("-"*40)
        
        f1_score = summary['avg_f1']
        if f1_score >= 0.8:
            grade = "🥇 Excellent"
        elif f1_score >= 0.7:
            grade = "🥈 Good"
        elif f1_score >= 0.6:
            grade = "🥉 Fair"
        else:
            grade = "❌ Needs improvement"