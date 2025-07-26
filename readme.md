# PDF Outline Extractor

A production-ready system for extracting hierarchical outlines (titles, headings, sub-headings) from complex PDF documents using hybrid ML/rule-based approaches.

## Improvised large results (Best score so far, after updated_processor.py):

```bash
adobe) santhankumar@fedora:~/Desktop/notebooks/Adobe Hackathon$ python new_validate.py ./output ./ground_truths
2025-07-25 23:13:45,631 - INFO - Found 5 ground truth files. Starting validation...
================================================================================
=== PDF STRUCTURE EXTRACTION VALIDATION REPORT ===
================================================================================

--- OVERALL SUMMARY ---
Files Processed: 5
Title Accuracy: 40.00%
Heading Precision: 57.69%
Heading Recall: 77.59%
Heading F1-Score: 66.18

--- PER-FILE PERFORMANCE ---
Filename                  | Title Match  | Precision  | Recall     | F1-Score  
-------------------------------------------------------------------------------
file01.json               | ✗            | 0.00       | 0.00       | 0.00      
file02.json               | ✗            | 68.18      | 88.24      | 76.92     
file03.json               | ✗            | 63.83      | 76.92      | 69.77     
file04.json               | ✓            | 0.00       | 0.00       | 0.00      
file05.json               | ✓            | 0.00       | 0.00       | 0.00      

--- ERROR ANALYSIS & RECOMMENDATIONS ---

Analysis of 13 Missed Headings (False Negatives):
  - Average Length: 4.4 words / 30.2 characters.
  - Observed Patterns:
    - All Caps: 1 (7.7%)
  > Recommendation: To improve recall, check if the extractor is missing headings with these patterns.

Analysis of 33 Incorrectly Identified Headings (False Positives):
  - Average Length: 6.2 words / 42.0 characters.
  - Observed Patterns:
    - Numbered: 13 (39.4%)
    - Ends With Punctuation: 4 (12.1%)
    - All Caps: 4 (12.1%)
    - Starts Lowercase: 1 (3.0%)
    - Decimal: 5 (15.2%)
  > Recommendation: To improve precision, consider adding filters to penalize or ignore text with these patterns.

Analysis of 20 Level Mismatches:
  - Level 2 was misclassified as 3: 11 time(s)
  - Level 1 was misclassified as 2: 4 time(s)
  - Level 4 was misclassified as 3: 4 time(s)
  - Level 1 was misclassified as 3: 1 time(s)
  > Recommendation: Review the logic for assigning heading levels, especially for these common misclassifications.

================================================================================
(adobe) santhankumar@fedora:~/Desktop/notebooks/Adobe Hackathon$ python validation.py ./output ./ground_truths

========================================
=== FILE-BY-FILE ANALYSIS ===
========================================

--- File: file01.json ---
  Title Match: ✗ (Similarity: 26.54%)
  Heading Counts:
    - Correctly Found (True Positives):  0
    - Incorrectly Found (False Positives): 6
    - Missed (False Negatives):          0
  Metrics:
    - Precision: 0.00%
    - Recall:    0.00%
    - F1-Score:  0.00%
  Insights:
    • Title partially matched with 26.54% similarity
    • High False Positive Rate: The model is over-detecting headings (6 incorrect vs. 0 missed).
  Recommendations for Improvement:
    • Consider increasing the heading score threshold to reduce false positives.

--- File: file02.json ---
  Title Match: ✗ (Similarity: 36.36%)
  Heading Counts:
    - Correctly Found (True Positives):  14
    - Incorrectly Found (False Positives): 8
    - Missed (False Negatives):          3
  Metrics:
    - Precision: 63.64%
    - Recall:    82.35%
    - F1-Score:  71.79%
  Insights:
    • Title partially matched with 36.36% similarity
    • High False Positive Rate: The model is over-detecting headings (8 incorrect vs. 3 missed).
    • Level Assignment Issue: 12 headings are found but assigned the wrong level.
  Analysis of 3 Missed Headings:
    - Average Length: 33.3 chars, 4.3 words
    - Common patterns observed in missed headings:
      • Numbered: 1 instance(s)
      • Capitalized: 2 instance(s)
  Recommendations for Improvement:
    • Consider increasing the heading score threshold to reduce false positives.
    • Improve detection logic for numbered headings (e.g., '1.2 Section').
    • Refine heading level assignment logic, possibly using font size more aggressively.

--- File: file03.json ---
  Title Match: ✗ (Similarity: 19.05%)
  Heading Counts:
    - Correctly Found (True Positives):  29
    - Incorrectly Found (False Positives): 18
    - Missed (False Negatives):          10
  Metrics:
    - Precision: 61.70%
    - Recall:    74.36%
    - F1-Score:  67.44%
  Insights:
    • Title not correctly identified
    • High False Positive Rate: The model is over-detecting headings (18 incorrect vs. 10 missed).
    • Level Assignment Issue: 6 headings are found but assigned the wrong level.
  Analysis of 10 Missed Headings:
    - Average Length: 33.7 chars, 5.0 words
    - Common patterns observed in missed headings:
      • Capitalized: 10 instance(s)
  Recommendations for Improvement:
    • Consider increasing the heading score threshold to reduce false positives.
    • Refine heading level assignment logic, possibly using font size more aggressively.

--- File: file04.json ---
  Title Match: ✓ (Similarity: 100.00%)
  Heading Counts:
    - Correctly Found (True Positives):  0
    - Incorrectly Found (False Positives): 1
    - Missed (False Negatives):          1
  Metrics:
    - Precision: 0.00%
    - Recall:    0.00%
    - F1-Score:  0.00%
  Analysis of 1 Missed Headings:
    - Average Length: 15.0 chars, 2.0 words
    - Common patterns observed in missed headings:
      • All caps: 1 instance(s)
      • Capitalized: 1 instance(s)
  Recommendations for Improvement:
    • Enhance detection of ALL CAPS headings.

--- File: file05.json ---
  Title Match: ✓ (Similarity: 100.00%)
  Heading Counts:
    - Correctly Found (True Positives):  0
    - Incorrectly Found (False Positives): 2
    - Missed (False Negatives):          1
  Metrics:
    - Precision: 0.00%
    - Recall:    0.00%
    - F1-Score:  0.00%
  Insights:
    • High False Positive Rate: The model is over-detecting headings (2 incorrect vs. 1 missed).
  Analysis of 1 Missed Headings:
    - Average Length: 22.0 chars, 5.0 words
    - Common patterns observed in missed headings:
      • Capitalized: 1 instance(s)
  Recommendations for Improvement:
    • Consider increasing the heading score threshold to reduce false positives.


========================================
=== OVERALL PERFORMANCE & SUGGESTIONS ===
========================================

Processed 5 files.
Overall Title Accuracy: 40.00%
Overall Heading Metrics:
  - Precision: 55.13%
  - Recall:    74.14%
  - F1-Score:  63.24%

Key Areas for Algorithmic Improvement:
  • Consider increasing the heading score threshold to reduce false positives. (Suggested for 4 file(s))
  • Refine heading level assignment logic, possibly using font size more aggressively. (Suggested for 2 file(s))
  • Improve detection logic for numbered headings (e.g., '1.2 Section'). (Suggested for 1 file(s))
  • Enhance detection of ALL CAPS headings. (Suggested for 1 file(s))

========================================
(adobe) santhankumar@fedora:~/Desktop/notebooks/Adobe Hackathon$ 
```








```bash
(adobe) santhankumar@fedora:~/Desktop/notebooks/Adobe Hackathon$ python validate_output.py
Validating extracted PDF structure against gold standard...

=== Detailed Evaluation Results ===

--- File: file01.json ---
  Title:
    - Match: ✗ (similarity: 0.00)
  Headings:
    - Ground truth:     0
    - Extracted:        5
    - Correctly found:  0
    - False positives:  5
    - False negatives:  0
  Metrics:
    - Precision: 0.00%
    - Recall:    0.00%
    - F1-Score:  0.00%
  Insights:
    - Title was not detected correctly
    - Too many false positives: 5 detected vs 0 missed
  Example false positives:
  • "Application form for grant of LTC advance" (level: 3)
  • "PAY + SI + NPA" (level: 1)
  • "10." (level: 1)
  Recommendations:
    - Increase heading score threshold to reduce false positives

--- File: file02.json ---
  Title:
    - Match: ✓ (similarity: 1.00)
  Headings:
    - Ground truth:     17
    - Extracted:        0
    - Correctly found:  0
    - False positives:  0
    - False negatives:  17
  Metrics:
    - Precision: 0.00%
    - Recall:    0.00%
    - F1-Score:  0.00%
  Insights:
    - Significant headings missed: 17 missed vs 0 found
    - Missing many decimal-numbered headings (like 1.2)
  Example missed headings:
  • "Revision History " (level: H1)
  • "Table of Contents " (level: H1)
  • "Acknowledgements " (level: H1)
  Recommendations:
    - Improve detection of numbered headings

--- File: file03.json ---
  Title:
    - Match: ✓ (similarity: 1.00)
  Headings:
    - Ground truth:     39
    - Extracted:        68
    - Correctly found:  33
    - False positives:  35
    - False negatives:  6
  Metrics:
    - Precision: 48.53%
    - Recall:    84.62%
    - F1-Score:  61.68%
  Insights:
    - Too many false positives: 35 detected vs 6 missed
    - 33 headings matched text but had wrong level
  Example false positives:
  • "Ontario’s Libraries" (level: 2)
  • "Working Together" (level: 2)
  • "RFP: R" (level: 1)
  Example missed headings:
  • "A Critical Component for Implementing Ontario’s Ro..." (level: H1)
  • "For the Ontario government it could mean: " (level: H4)
  • "Appendix A: ODL Envisioned Phases & Funding " (level: H2)
  Recommendations:
    - Increase heading score threshold to reduce false positives
    - Refine heading level assignment logic

--- File: file04.json ---
  Title:
    - Match: ✓ (similarity: 1.00)
  Headings:
    - Ground truth:     1
    - Extracted:        6
    - Correctly found:  1
    - False positives:  5
    - False negatives:  0
  Metrics:
    - Precision: 16.67%
    - Recall:    100.00%
    - F1-Score:  28.57%
  Insights:
    - Too many false positives: 5 detected vs 0 missed
    - 1 headings matched text but had wrong level
  Example false positives:
  • "Parsippany -Troy Hills STEM Pathways" (level: 1)
  • "Mission Statement: To provide PTHSD high school st..." (level: 2)
  • "Goals:" (level: 3)
  Recommendations:
    - Increase heading score threshold to reduce false positives
    - Refine heading level assignment logic

--- File: file05.json ---
  Title:
    - Match: ✓ (similarity: 1.00)
  Headings:
    - Ground truth:     1
    - Extracted:        10
    - Correctly found:  0
    - False positives:  10
    - False negatives:  1
  Metrics:
    - Precision: 0.00%
    - Recall:    0.00%
    - F1-Score:  0.00%
  Insights:
    - Too many false positives: 10 detected vs 1 missed
  Example false positives:
  • "ADDRESS:" (level: 1)
  • "TOPJUMP" (level: 1)
  • "3735 PARKWAY" (level: 1)
  Example missed headings:
  • "HOPE To SEE You THERE! " (level: H1)
  Recommendations:
    - Increase heading score threshold to reduce false positives


==========================
=== EVALUATION SUMMARY ===
==========================

Total Files Evaluated: 5
Title Accuracy: 80.00%

--- Outline Heading Performance ---
  - Precision: 38.20%
  - Recall:    58.62%
  - F1-Score:  46.26%
  (34 correct out of 58 ground truth headings)

--- Global Patterns & Issues ---
  • GLOBAL ISSUE: Model is over-detecting, leading to high false positives.
  • PATTERN: Model is commonly missing numbered headings (e.g., '1.2 Section').

--- Suggested Optimizations ---
  • 1. High False Positives: Increase heading score threshold to be more strict.
  • 3. Missing Numbered Headings: Improve regex for numbered patterns.
  • 4. Level Mismatches: Refine logic for assigning heading levels (e.g., use font size more effectively).

==========================
```



## 🏗️ Architecture

```
PDF Input → Layout Analysis → Feature Extraction → Classification → Post-processing → JSON Output
           (PyMuPDF)       (Geometry + MiniLM)   (MLP)          (Hierarchy Repair)
```

### Model Pipeline:
1. **Text Extraction**: Extract spans with font, position, styling metadata
2. **Feature Engineering**: 391-dim vectors (384 semantic + 7 geometric)
3. **Classification**: 4-class MLP (BODY, H1, H2, H3)
4. **Post-processing**: Merge multi-line headings, filter false positives, repair hierarchy

## 📁 Project Structure

```
project_root/
├── Dockerfile                 # Production container
├── requirements.txt          # Python dependencies
├── build.py                  # Automated build script
├── README.md                # This file
├── models/                   # Trained models (created during build)
│   ├── minilm_quantized/    # Quantized sentence transformer
│   ├── mlp_head.pt          # Trained MLP classifier
│   └── model_metadata.json  # Model configuration
├── src/                      # Source code
│   ├── prepare_models.py    # Model training and quantization
│   ├── pdf_outline.py       # Main extraction pipeline
│   └── validate_output.py   # Output validation and metrics
└── proto_labels/            # Training data (hand-labeled examples)
    └── sample_labels.json   # Example labeled data
```

## 🚀 Quick Start

### 1. Build System

```bash
# Automated build (recommended)
python build.py

# Manual build
pip install -r requirements.txt
python src/prepare_models.py
```

### 2. Extract Outlines

```bash
# Single PDF
python src/pdf_outline.py --single document.pdf --output results/

# Directory of PDFs  
python src/pdf_outline.py --input pdfs/ --output results/

# Docker (production)
docker build --platform linux/amd64 -t pdf-extractor .
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  pdf-extractor
```

### 3. Validate Results

```bash
# Compare against ground truth
python src/validate_output.py \
  --predicted results/ \
  --ground-truth ground_truth/ 

# Single file validation
python src/validate_output.py \
  --single-pred results/doc.json \
  --single-gt ground_truth/doc.json
```

## 📋 Requirements

### System Requirements
- **Platform**: Linux AMD64 (Docker) or compatible
- **Memory**: 16GB RAM recommended, 8GB minimum
- **CPU**: 8 cores recommended for optimal performance
- **Storage**: 500MB for models and dependencies

### Python Dependencies
- Python 3.10+
- PyMuPDF 1.23.26+ (PDF processing)
- PyTorch 2.1.0+ (neural networks)
- sentence-transformers 2.2.2+ (embeddings)
- scikit-learn 1.3+ (utilities)
- NumPy, SciPy (numerical computing)

## 🔧 Configuration

### Model Training

Create `proto_labels/your_labels.json`:

```json
{
  "pdf": "sample_document.pdf",
  "labeled_lines": [
    {"text": "1. Introduction", "label": "H1"},
    {"text": "1.1 Background", "label": "H2"},
    {"text": "1.1.1 Motivation", "label": "H3"},
    {"text": "This document presents...", "label": "BODY"}
  ]
}
```

### Output Format

Generated JSON structure:

```json
{
  "title": "Document Title",
  "outline": [
    {"level": "H1", "text": "Introduction", "page": 1},
    {"level": "H2", "text": "Background", "page": 2},
    {"level": "H3", "text": "Methodology", "page": 3}
  ]
}
```

## 🧪 Testing & Validation

### Automated Testing

```bash
# Run full validation suite
python src/validate_output.py --predicted output/ --ground-truth truth/

# Performance benchmarking
time python src/pdf_outline.py --single large_document.pdf
```

### Manual Quality Checks

```bash
# Check model size
du -sh models/
# Should be ≤200MB

# Test edge cases
python src/pdf_outline.py --single multilingual.pdf
python src/pdf_outline.py --single complex_tables.pdf
```

### Expected Performance
- **Accuracy**: ≥80% title detection, ≥70% outline F1-score
- **Speed**: ≤10 seconds for 50-page PDFs
- **Memory**: ≤3GB peak usage
- **Model Size**: ≤200MB total

## 🔍 Advanced Usage

### Custom Model Training

```bash
# Train with your labeled data
python src/prepare_models.py \
  --proto-labels your_labels/ \
  --model-dir custom_models/

# Use custom models
python src/pdf_outline.py \
  --models custom_models/ \
  --input pdfs/ \
  --output results/
```

### Extending to More Heading Levels

1. **Update Labels**: Add H4, H5 to `label_to_idx` in `prepare_models.py`
2. **Retrain MLP**: Increase `num_classes` parameter
3. **Update Post-processing**: Extend hierarchy rules in `OutlinePostProcessor`

### Adapting to New Schemas

Modify the output format in `pdf_outline.py`:

```python
# Current format
outline.append({
    "level": span.get("pred", "H1"),
    "text": span.get("text", "").strip(),
    "page": span.get("page", 0) + 1
})

# Custom format example
outline.append({
    "heading_type": span.get("pred", "H1"),
    "content": span.get("text", "").strip(),
    "page_number": span.get("page", 0) + 1,
    "confidence": span.get("confidence", 0.0),
    "bbox": [span.get("x0"), span.get("y0"), span.get("x1"), span.get("y1")]
})
```

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Rebuild models
rm -rf models/
python src/prepare_models.py
```

**Memory Issues**
```bash
# Reduce batch size in pdf_outline.py
embeddings = self.encoder.encode(texts, batch_size=32)  # Reduce from 64
```

**Poor Accuracy**
```bash
# Add more training data
# Check proto_labels/ directory
# Verify ground truth quality
python src/validate_output.py --predicted out/ --ground-truth truth/
```

**Docker Build Fails**
```bash
# Check platform
docker build --platform linux/amd64 -t pdf-extractor .

# Check dependencies
pip install -r requirements.txt
```

### Performance Optimization

**Speed Improvements:**
- Reduce MLP hidden dimension (128 → 64)
- Lower embedding batch size for memory-constrained systems
- Skip very short text spans (< 3 characters)

**Accuracy Improvements:**
- Add domain-specific training data
- Tune similarity thresholds in post-processing
- Adjust font size percentile thresholds


### Training Data Guidelines

- **Balanced**: ~70 examples per class (H1, H2, H3, BODY)
- **Diverse**: Multiple document types and languages
- **Clean**: Accurate labels, consistent formatting
- **Representative**: Cover edge cases (dates, tables, multilingual)

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **PyMuPDF**: High-performance PDF processing
- **Sentence Transformers**: Efficient text embeddings
- **PyTorch**: Neural network framework
- **scikit-learn**: Machine learning utilities

## 📈 Roadmap

### Version 2.0 (Planned)
- [ ] Support for H4, H5+ heading levels
- [ ] Table of contents detection and linking
- [ ] Multi-column layout handling
- [ ] Figure/table caption extraction
- [ ] Cross-reference resolutio
## 🔗 Related Projects

- [Adobe PDF Embed API](https://developer.adobe.com/document-services/apis/pdf-embed/) - For frontend integration
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/) - PDF processing reference
- [Sentence Transformers](https://www.sbert.net/) - Semantic embeddings

---

**Built for the "Connecting the Dots" Hackathon** 🏆

For questions, issues, or contributions, please open an issue on the repository.
