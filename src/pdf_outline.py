#!/usr/bin/env python3
"""
Production PDF outline extraction system.

This system extracts hierarchical outlines from PDF documents using:
1. Layout analysis with PyMuPDF
2. Semantic embeddings with quantized MiniLM
3. MLP classification for heading detection
4. Post-processing for hierarchy repair and noise removal
"""

import os
import sys
import json
import re
import time
import argparse
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from pathlib import Path

import fitz  # PyMuPDF
import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist


class MLPHeadClassifier(nn.Module):
    """Lightweight MLP for heading classification."""
    
    def __init__(self, input_dim: int = 391, hidden_dim: int = 128, num_classes: int = 4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class GeometryFeatureExtractor:
    """Extracts geometric features from text spans."""
    
    @staticmethod
    def z_normalize(values: np.ndarray) -> np.ndarray:
        """Z-score normalization with zero-division protection."""
        if len(values) == 0:
            return values
        std = values.std()
        if std == 0:
            return np.zeros_like(values)
        return (values - values.mean()) / std
    
    @staticmethod
    def extract_font_features(spans: List[Dict]) -> Dict[str, Any]:
        """Extract font-based features from spans."""
        if not spans:
            return {}
        
        font_sizes = [span.get("font_size", 12) for span in spans]
        font_sizes_array = np.array(font_sizes)
        
        return {
            "median_font_size": np.median(font_sizes_array),
            "font_size_std": font_sizes_array.std(),
            "font_size_range": font_sizes_array.max() - font_sizes_array.min(),
            "normalized_sizes": GeometryFeatureExtractor.z_normalize(font_sizes_array)
        }
    
    @staticmethod
    def detect_body_margin(spans: List[Dict], font_stats: Dict) -> float:
        """Detect the most common left margin for body text."""
        if not spans or not font_stats:
            return 0.0
        
        median_size = font_stats.get("median_font_size", 12)
        tolerance = 0.5
        
        # Find spans with font size close to median (likely body text)
        body_margins = []
        for span in spans:
            if abs(span.get("font_size", 12) - median_size) < tolerance:
                body_margins.append(round(span.get("x0", 0)))
        
        if not body_margins:
            return 0.0
        
        # Return most common margin
        margin_counter = Counter(body_margins)
        return margin_counter.most_common(1)[0][0]
    
    @staticmethod
    def extract_geometry_vector(span: Dict, page_width: float, page_height: float, 
                              normalized_font: float, body_margin: float) -> np.ndarray:
        """Extract 7-dimensional geometry feature vector."""
        x0, y0, x1, y1 = span.get("x0", 0), span.get("y0", 0), span.get("x1", 0), span.get("y1", 0)
        
        # Normalized positions and dimensions
        norm_x = x0 / page_width if page_width > 0 else 0
        norm_y = y0 / page_height if page_height > 0 else 0
        norm_width = (x1 - x0) / page_width if page_width > 0 else 0
        
        # Font and style features
        is_bold = span.get("bold", False)
        is_italic = span.get("italic", False)
        
        # Indentation feature (distance from body margin)
        is_indented = int(abs(x0 - body_margin) > 20) if body_margin > 0 else 0
        
        return np.array([
            norm_x,           # Normalized x position
            norm_y,           # Normalized y position  
            norm_width,       # Normalized width
            normalized_font,  # Z-normalized font size
            int(is_bold),     # Bold flag
            int(is_italic),   # Italic flag
            is_indented       # Indentation flag
        ], dtype=np.float32)


class TextNormalizer:
    """Handles text cleaning and normalization."""
    
    # Regex patterns for common non-heading patterns
    DATE_PATTERN = re.compile(
        r'\b\d{1,2}\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4}\b', 
        re.IGNORECASE
    )
    VERSION_PATTERN = re.compile(r'^\d+(\.\d+){0,3}$')
    PAGE_NUMBER_PATTERN = re.compile(r'^(page\s+)?\d+(\s+of\s+\d+)?$', re.IGNORECASE)
    TOC_PATTERN = re.compile(r'table\s+of\s+contents', re.IGNORECASE)
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove non-printable characters except common punctuation
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        return text
    
    @staticmethod
    def looks_like_table_content(text: str, word_count: int, font_size_ratio: float) -> bool:
        """Detect if text looks like table content rather than heading."""
        if not text:
            return True
        
        text_clean = text.strip()
        
        # Short numeric or date patterns
        if TextNormalizer.DATE_PATTERN.fullmatch(text_clean):
            return True
        
        if TextNormalizer.VERSION_PATTERN.fullmatch(text_clean):
            return True
        
        if TextNormalizer.PAGE_NUMBER_PATTERN.fullmatch(text_clean):
            return True
        
        # Simple digit-only content
        if text_clean.isdigit():
            return True
        
        # Very small font relative to median
        if font_size_ratio < 0.2:
            return True
        
        # Table of contents entries
        if TextNormalizer.TOC_PATTERN.search(text_clean):
            return True
        
        # Very short content with few words (likely table cells)
        if word_count <= 2 and len(text_clean) < 10:
            return True
        
        return False
    
    @staticmethod
    def is_likely_heading(text: str) -> bool:
        """Basic heuristics to identify potential headings."""
        if not text:
            return False
        
        text_clean = text.strip()
        
        # Numbered sections (e.g., "1.1", "2.3.4")
        if re.match(r'^\d+(\.\d+)*\s+', text_clean):
            return True
        
        # Chapter/Section patterns
        if re.match(r'^(chapter|section|part|appendix)\s+\d+', text_clean, re.IGNORECASE):
            return True
        
        # Title case patterns
        if text_clean.istitle() and len(text_clean.split()) >= 2:
            return True
        
        # All caps (but not too long to avoid false positives)
        if text_clean.isupper() and 3 <= len(text_clean) <= 50:
            return True
        
        return False


class OutlinePostProcessor:
    """Handles post-processing and hierarchy repair."""
    
    def __init__(self):
        self.labels = ["BODY", "H1", "H2", "H3"]
    
    def merge_multiline_headings(self, spans: List[Dict]) -> List[Dict]:
        """Merge headings that are split across multiple lines."""
        if not spans:
            return spans
        
        # Sort by page and y-position
        sorted_spans = sorted(spans, key=lambda x: (x.get("page", 0), x.get("y0", 0)))
        
        merged = []
        i = 0
        
        while i < len(sorted_spans):
            current = sorted_spans[i].copy()
            
            # Skip body text
            if current.get("pred", "BODY") == "BODY":
                i += 1
                continue
            
            # Look for continuation lines
            j = i + 1
            while j < len(sorted_spans):
                next_span = sorted_spans[j]
                
                # Same page, same prediction level, close vertical distance
                if (next_span.get("page") == current.get("page") and
                    next_span.get("pred") == current.get("pred") and
                    abs(next_span.get("y0", 0) - current.get("y1", 0)) < 6 and
                    abs(next_span.get("x0", 0) - current.get("x0", 0)) < 5):
                    
                    # Merge text and update bounding box
                    current["text"] += " " + next_span.get("text", "")
                    current["y1"] = next_span.get("y1", current["y1"])
                    current["x1"] = max(current.get("x1", 0), next_span.get("x1", 0))
                    j += 1
                else:
                    break
            
            merged.append(current)
            i = j
        
        return merged
    
    def filter_false_positives(self, spans: List[Dict], font_stats: Dict) -> List[Dict]:
        """Remove false positive headings (dates, page numbers, etc.)."""
        if not spans or not font_stats:
            return spans
        
        median_font = font_stats.get("median_font_size", 12)
        filtered = []
        
        for span in spans:
            text = span.get("text", "").strip()
            font_size = span.get("font_size", 12)
            word_count = len(text.split())
            font_ratio = font_size / median_font if median_font > 0 else 1.0
            
            # Skip if looks like table content
            if TextNormalizer.looks_like_table_content(text, word_count, font_ratio):
                continue
            
            # Skip very short headings unless they have strong indicators
            if len(text) < 3 and not TextNormalizer.is_likely_heading(text):
                continue
            
            # Skip if confidence is very low
            if span.get("confidence", 0.0) < 0.3:
                continue
            
            filtered.append(span)
        
        return filtered
    
    def repair_hierarchy(self, spans: List[Dict]) -> List[Dict]:
        """Repair heading hierarchy based on font sizes and nesting rules."""
        if not spans:
            return spans
        
        # Group by font characteristics for level assignment
        font_groups = defaultdict(list)
        for span in spans:
            font_key = (round(span.get("font_size", 12), 1), span.get("bold", False))
            font_groups[font_key].append(span)
        
        # Sort font groups by size (descending) and bold status
        sorted_groups = sorted(
            font_groups.keys(),
            key=lambda x: (x[0], x[1]),  # font_size, is_bold
            reverse=True
        )
        
        # Assign levels based on font hierarchy
        level_mapping = {}
        for i, font_key in enumerate(sorted_groups[:3]):  # Max 3 heading levels
            level_mapping[font_key] = f"H{i + 1}"
        
        # Apply level corrections
        corrected = []
        page_h1_seen = {}
        page_h2_seen = {}
        
        for span in spans:
            font_key = (round(span.get("font_size", 12), 1), span.get("bold", False))
            
            # Get corrected level
            corrected_level = level_mapping.get(font_key, span.get("pred", "H1"))
            page_num = span.get("page", 0)
            
            # Enforce hierarchy rules: H2 requires H1, H3 requires H2
            if corrected_level == "H2" and not page_h1_seen.get(page_num, False):
                corrected_level = "H1"
                page_h1_seen[page_num] = True
            elif corrected_level == "H3" and not page_h2_seen.get(page_num, False):
                corrected_level = "H2"
                page_h2_seen[page_num] = True
            
            # Update tracking
            if corrected_level == "H1":
                page_h1_seen[page_num] = True
                page_h2_seen[page_num] = False  # Reset H2 tracking
            elif corrected_level == "H2":
                page_h2_seen[page_num] = True
            
            # Create corrected span
            corrected_span = span.copy()
            corrected_span["pred"] = corrected_level
            corrected.append(corrected_span)
        
        return corrected
    
    def remove_duplicates(self, spans: List[Dict]) -> List[Dict]:
        """Remove duplicate headings based on text similarity."""
        if len(spans) <= 1:
            return spans
        
        # Extract text for similarity comparison
        texts = [span.get("text", "") for span in spans]
        
        # Simple deduplication based on exact text match
        seen_texts = set()
        deduplicated = []
        
        for span in spans:
            text = span.get("text", "").strip().lower()
            if text not in seen_texts:
                seen_texts.add(text)
                deduplicated.append(span)
        
        return deduplicated


class PDFOutlineExtractor:
    """Main PDF outline extraction system."""
    
    def __init__(self, model_dir: str = "/app/models"):
        self.model_dir = model_dir
        self.encoder = None
        self.mlp = None
        self.metadata = None
        self.geometry_extractor = GeometryFeatureExtractor()
        self.text_normalizer = TextNormalizer()
        self.post_processor = OutlinePostProcessor()
        
        self._load_models()
    
    def _load_models(self):
        """Load quantized MiniLM and trained MLP classifier."""
        try:
            # Load quantized sentence transformer
            minilm_path = os.path.join(self.model_dir, "minilm_quantized")
            if os.path.exists(minilm_path):
                print(f"📥 Loading quantized MiniLM from {minilm_path}")
                self.encoder = SentenceTransformer(minilm_path)
            else:
                raise FileNotFoundError(f"Quantized MiniLM not found at {minilm_path}")
            
            # Load model metadata
            metadata_path = os.path.join(self.model_dir, "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                # Default metadata if file not found
                self.metadata = {
                    'label_to_idx': {"BODY": 0, "H1": 1, "H2": 2, "H3": 3},
                    'idx_to_label': {0: "BODY", 1: "H1", 2: "H2", 3: "H3"},
                    'input_dim': 391,
                    'hidden_dim': 128,
                    'num_classes': 4
                }
            
            # Load MLP classifier
            mlp_path = os.path.join(self.model_dir, "mlp_head.pt")
            if os.path.exists(mlp_path):
                print(f"📥 Loading MLP classifier from {mlp_path}")
                self.mlp = MLPHeadClassifier(
                    input_dim=self.metadata['input_dim'],
                    hidden_dim=self.metadata['hidden_dim'],
                    num_classes=self.metadata['num_classes']
                )
                self.mlp.load_state_dict(torch.load(mlp_path, map_location='cpu'))
                self.mlp.eval()
            else:
                raise FileNotFoundError(f"MLP classifier not found at {mlp_path}")
            
            print("✅ Models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            sys.exit(1)
    
    def extract_text_spans(self, pdf_path: str) -> List[Dict]:
        """Extract text spans from PDF with layout information."""
        try:
            doc = fitz.open(pdf_path)
            all_spans = []
            
            for page_idx, page in enumerate(doc):
                # Get text blocks with detailed formatting
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if block.get("type") != 0:  # Skip non-text blocks
                        continue
                    
                    for line in block.get("lines", []):
                        line_text = ""
                        line_spans = line.get("spans", [])
                        
                        if not line_spans:
                            continue
                        
                        # Combine span texts
                        for span in line_spans:
                            line_text += span.get("text", "")
                        
                        line_text = self.text_normalizer.clean_text(line_text)
                        if not line_text:
                            continue
                        
                        # Use first span for formatting info
                        first_span = line_spans[0]
                        flags = first_span.get("flags", 0)
                        
                        span_info = {
                            "text": line_text,
                            "page": page_idx,
                            "font_size": first_span.get("size", 12),
                            "bold": bool(flags & 2**4),  # Bold flag
                            "italic": bool(flags & 2**1),  # Italic flag
                            "x0": line.get("bbox", [0, 0, 0, 0])[0],
                            "y0": line.get("bbox", [0, 0, 0, 0])[1],
                            "x1": line.get("bbox", [0, 0, 0, 0])[2],
                            "y1": line.get("bbox", [0, 0, 0, 0])[3],
                            "page_width": page.rect.width,
                            "page_height": page.rect.height
                        }
                        
                        all_spans.append(span_info)
            
            doc.close()
            return all_spans
            
        except Exception as e:
            print(f"❌ Error extracting text spans from {pdf_path}: {e}")
            return []
    
    def classify_spans(self, spans: List[Dict]) -> List[Dict]:
        """Classify text spans as headings or body text."""
        if not spans:
            return spans
        
        # Extract font statistics
        font_stats = self.geometry_extractor.extract_font_features(spans)
        
        # Detect body text margin
        body_margin = self.geometry_extractor.detect_body_margin(spans, font_stats)
        
        # Encode text content
        texts = [span["text"] for span in spans]
        print(f"🔄 Encoding {len(texts)} text spans...")
        
        try:
            embeddings = self.encoder.encode(
                texts, 
                batch_size=64, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
        except Exception as e:
            print(f"❌ Error encoding text: {e}")
            return spans
        
        # Extract geometry features and combine with embeddings
        features = []
        normalized_sizes = font_stats.get("normalized_sizes", np.zeros(len(spans)))
        
        for i, span in enumerate(spans):
            # Get normalized font size for this span
            norm_font = normalized_sizes[i] if i < len(normalized_sizes) else 0.0
            
            # Extract geometry vector
            geom_vector = self.geometry_extractor.extract_geometry_vector(
                span, 
                span.get("page_width", 1),
                span.get("page_height", 1),
                norm_font,
                body_margin
            )
            
            # Combine embedding with geometry features
            combined_features = np.concatenate([embeddings[i], geom_vector])
            features.append(combined_features)
        
        # Classify with MLP
        features_tensor = torch.FloatTensor(np.vstack(features))
        
        with torch.no_grad():
            logits = self.mlp(features_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
        
        # Add predictions to spans
        idx_to_label = self.metadata["idx_to_label"]
        for i, span in enumerate(spans):
            pred_idx = predictions[i].item()
            span["pred"] = idx_to_label[pred_idx]
            span["confidence"] = confidences[i].item()
        
        return spans
    
    def detect_title(self, spans: List[Dict], pdf_path: str) -> str:
        """Detect document title from first page."""
        if not spans:
            return os.path.basename(pdf_path).replace('.pdf', '')
        
        # Look for title candidates on first page
        first_page_spans = [s for s in spans if s.get("page", 0) == 0]
        
        if not first_page_spans:
            return os.path.basename(pdf_path).replace('.pdf', '')
        
        # Get font size statistics for first page
        font_sizes = [s.get("font_size", 12) for s in first_page_spans]
        if not font_sizes:
            return os.path.basename(pdf_path).replace('.pdf', '')
        
        font_85th = np.percentile(font_sizes, 85)
        
        # Find title candidates: large font, bold, near top of page
        title_candidates = []
        for span in first_page_spans:
            if (span.get("font_size", 12) >= font_85th and
                span.get("bold", False) and
                span.get("y0", float('inf')) < 200):  # Near top of page
                title_candidates.append(span)
        
        if not title_candidates:
            return os.path.basename(pdf_path).replace('.pdf', '')
        
        # Select title: largest font, then highest on page
        title_span = max(
            title_candidates,
            key=lambda s: (s.get("font_size", 12), -s.get("y0", 0))
        )
        
        return self.text_normalizer.clean_text(title_span["text"])
    
    def extract_outline(self, pdf_path: str) -> Dict[str, Any]:
        """Extract complete outline from PDF."""
        print(f"📄 Processing {os.path.basename(pdf_path)}...")
        
        start_time = time.time()
        
        # Step 1: Extract text spans
        spans = self.extract_text_spans(pdf_path)
        if not spans:
            return {"title": os.path.basename(pdf_path).replace('.pdf', ''), "outline": []}
        
        print(f"📊 Extracted {len(spans)} text spans")
        
        # Step 2: Classify spans
        classified_spans = self.classify_spans(spans)
        
        # Step 3: Filter to heading candidates
        heading_spans = [s for s in classified_spans if s.get("pred", "BODY") != "BODY"]
        print(f"📊 Found {len(heading_spans)} heading candidates")
        
        # Step 4: Post-processing
        if heading_spans:
            # Merge multi-line headings
            merged_spans = self.post_processor.merge_multiline_headings(heading_spans)
            print(f"📊 After merging: {len(merged_spans)} headings")
            
            # Filter false positives
            font_stats = self.geometry_extractor.extract_font_features(spans)
            filtered_spans = self.post_processor.filter_false_positives(merged_spans, font_stats)
            print(f"📊 After filtering: {len(filtered_spans)} headings")
            
            # Repair hierarchy
            repaired_spans = self.post_processor.repair_hierarchy(filtered_spans)
            
            # Remove duplicates
            final_spans = self.post_processor.remove_duplicates(repaired_spans)
            print(f"📊 Final headings: {len(final_spans)}")
        else:
            final_spans = []
        
        # Step 5: Detect title
        title = self.detect_title(spans, pdf_path)
        
        # Step 6: Format output
        outline = []
        for span in final_spans:
            outline.append({
                "level": span.get("pred", "H1"),
                "text": span.get("text", "").strip(),
                "page": span.get("page", 0) + 1  # Convert to 1-indexed
            })
        
        # Sort by page and position
        outline.sort(key=lambda x: (x["page"], x.get("y0", 0)))
        
        processing_time = time.time() - start_time
        print(f"⏱️  Processing completed in {processing_time:.2f} seconds")
        
        return {
            "title": title.strip(),
            "outline": outline
        }


def process_directory(input_dir: str, output_dir: str, model_dir: str):
    """Process all PDFs in input directory."""
    print(f"🚀 Starting PDF outline extraction")
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📂 Model directory: {model_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize extractor
    extractor = PDFOutlineExtractor(model_dir)
    
    # Find PDF files
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in input directory")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    total_start_time = time.time()
    successful = 0
    
    for pdf_file in pdf_files:
        try:
            pdf_path = os.path.join(input_dir, pdf_file)
            output_file = pdf_file.replace('.pdf', '.json')
            output_path = os.path.join(output_dir, output_file)
            
            # Extract outline
            result = extractor.extract_outline(pdf_path)
            
            # Save result
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✅ {pdf_file} -> {output_file}")
            successful += 1
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")
            continue
    
    total_time = time.time() - total_start_time
    print(f"\n🎉 Processing complete!")
    print(f"📊 Successfully processed: {successful}/{len(pdf_files)} files")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"⏱️  Average time per file: {total_time/len(pdf_files):.2f} seconds")


def main():
    """Main entry point for Docker container."""
    parser = argparse.ArgumentParser(description="Extract outlines from PDF documents")
    parser.add_argument("--input", default="/app/input", help="Input directory containing PDFs")
    parser.add_argument("--output", default="/app/output", help="Output directory for JSON files")
    parser.add_argument("--models", default="/app/models", help="Directory containing trained models")
    parser.add_argument("--single", help="Process single PDF file")
    
    args = parser.parse_args()
    
    if args.single:
        # Process single file
        extractor = PDFOutlineExtractor(args.models)
        result = extractor.extract_outline(args.single)
        
        output_file = os.path.basename(args.single).replace('.pdf', '.json')
        output_path = os.path.join(args.output, output_file)
        
        os.makedirs(args.output, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Result saved to {output_path}")
    else:
        # Process directory
        process_directory(args.input, args.output, args.models)


if __name__ == "__main__":
    main()