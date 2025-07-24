#!/usr/bin/env python3
"""
Model preparation script for PDF outline extraction.

This script handles:
1. Downloading and quantizing MiniLM-L6 model
2. Training MLP classifier on prototype corpus
3. Saving optimized models for production use
"""

import os
import json
import pickle
import argparse
from typing import List, Tuple, Dict, Any
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


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


class ModelPreparer:
    """Handles model downloading, quantization, and training."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Label mappings
        self.label_to_idx = {"BODY": 0, "H1": 1, "H2": 2, "H3": 3}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
    
    def quantize_minilm(self) -> str:
        """Download and quantize MiniLM model for deployment."""
        print("🔄 Downloading MiniLM-L6-v2...")
        
        # Download base model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        print("🔄 Quantizing model...")
        
        # Extract transformer module for quantization
        transformer_module = model._modules["0"]
        
        # Apply dynamic INT8 quantization
        quantized_transformer = torch.quantization.quantize_dynamic(
            transformer_module, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        
        # Create quantized model
        quantized_model = SentenceTransformer(modules=[quantized_transformer])
        
        # Save quantized model
        quantized_path = os.path.join(self.model_dir, "minilm_quantized")
        quantized_model.save(quantized_path)
        
        # Calculate size
        model_size_mb = sum(
            os.path.getsize(os.path.join(quantized_path, f)) 
            for f in os.listdir(quantized_path) if os.path.isfile(os.path.join(quantized_path, f))
        ) / (1024 * 1024)
        
        print(f"✅ Quantized model saved to {quantized_path} ({model_size_mb:.1f} MB)")
        return quantized_path
    
    def load_prototype_corpus(self, json_dir: str) -> Tuple[List[str], List[int]]:
        """Load hand-labeled prototype corpus from JSON files."""
        texts, labels = [], []
        
        if not os.path.exists(json_dir):
            print(f"⚠️  Prototype labels directory not found: {json_dir}")
            print("Creating minimal synthetic corpus for demonstration...")
            return self._create_synthetic_corpus()
        
        print(f"📁 Loading prototype corpus from {json_dir}")
        
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        if not json_files:
            print("⚠️  No JSON files found in prototype directory")
            return self._create_synthetic_corpus()
        
        for filename in json_files:
            filepath = os.path.join(json_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for line_data in data.get("labeled_lines", []):
                    text = line_data.get("text", "").strip()
                    label = line_data.get("label", "BODY")
                    
                    if text and label in self.label_to_idx:
                        texts.append(text)
                        labels.append(self.label_to_idx[label])
                
            except Exception as e:
                print(f"⚠️  Error loading {filename}: {e}")
                continue
        
        if not texts:
            print("⚠️  No valid labeled data found, creating synthetic corpus")
            return self._create_synthetic_corpus()
        
        label_counts = Counter(labels)
        print(f"📊 Loaded {len(texts)} labeled examples:")
        for label, count in label_counts.items():
            print(f"   {self.idx_to_label[label]}: {count}")
        
        return texts, labels
    
    def _create_synthetic_corpus(self) -> Tuple[List[str], List[int]]:
        """Create synthetic training corpus for demonstration."""
        print("🔧 Creating synthetic prototype corpus...")
        
        synthetic_data = [
            # H1 examples
            ("1. Introduction", 1),
            ("Chapter 1: Overview", 1),
            ("Executive Summary", 1),
            ("Background Information", 1),
            ("Project Description", 1),
            ("Methodology", 1),
            ("Literature Review", 1),
            ("Data Analysis", 1),
            ("Results and Discussion", 1),
            ("Conclusion", 1),
            
            # H2 examples  
            ("1.1 Objectives", 2),
            ("2.1 Research Questions", 2),
            ("3.1 Data Collection", 2),
            ("4.1 Statistical Methods", 2),
            ("5.1 Key Findings", 2),
            ("Problem Statement", 2),
            ("Scope and Limitations", 2),
            ("Technical Approach", 2),
            ("Implementation Details", 2),
            ("Performance Evaluation", 2),
            
            # H3 examples
            ("1.1.1 Primary Goals", 3),
            ("2.1.1 Survey Design", 3),
            ("3.1.1 Sample Selection", 3),
            ("4.1.1 Regression Analysis", 3),
            ("5.1.1 Quantitative Results", 3),
            ("Data Preprocessing", 3),
            ("Feature Selection", 3),
            ("Model Training", 3),
            ("Cross Validation", 3),
            ("Error Analysis", 3),
            
            # BODY examples
            ("This document presents the findings of our research study.", 0),
            ("The data was collected over a period of six months.", 0),
            ("Table 1 shows the demographic characteristics of participants.", 0),
            ("Figure 2 illustrates the correlation between variables.", 0),
            ("31 MAY 2014", 0),
            ("18 JUNE 2013", 0),
            ("Version 1.0", 0),
            ("Page 3 of 25", 0),
            ("For more information, contact the authors.", 0),
            ("Statistical significance was set at p < 0.05.", 0),
            ("The research was conducted in accordance with ethical guidelines.", 0),
            ("Participants were recruited through online advertisements.", 0),
            ("Data analysis was performed using SPSS version 28.", 0),
            ("The response rate was 73% (n=146).", 0),
            ("These findings are consistent with previous studies.", 0),
            ("Further research is needed to validate these results.", 0),
            ("The limitations of this study include sample size and scope.", 0),
            ("We thank the participants for their valuable contribution.", 0),
            ("All procedures were approved by the institutional review board.", 0),
            ("Copyright © 2024 International Research Foundation.", 0),
        ]
        
        texts, labels = zip(*synthetic_data)
        return list(texts), list(labels)
    
    def train_mlp_classifier(self, texts: List[str], labels: List[int], model_path: str) -> str:
        """Train MLP classifier on prototype corpus."""
        print("🔄 Loading quantized MiniLM for feature extraction...")
        
        # Load quantized model for encoding
        encoder = SentenceTransformer(model_path)
        
        print("🔄 Encoding text features...")
        embeddings = encoder.encode(texts, batch_size=32, convert_to_numpy=True)
        
        print(f"📊 Feature shape: {embeddings.shape}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Validation set: {len(X_val)} samples")
        
        # Initialize MLP (embedding dim = 384, add 7 for geometry features)
        mlp = MLPHeadClassifier(input_dim=384 + 7, hidden_dim=128, num_classes=4)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(mlp.parameters(), lr=4e-3, weight_decay=1e-4)
        
        # Prepare training data (pad with zeros for geometry features during training)
        X_train_padded = np.pad(X_train, ((0, 0), (0, 7)), mode='constant')
        X_val_padded = np.pad(X_val, ((0, 0), (0, 7)), mode='constant')
        
        X_train_tensor = torch.FloatTensor(X_train_padded)
        X_val_tensor = torch.FloatTensor(X_val_padded)
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        
        print("🔄 Training MLP classifier...")
        
        # Training loop
        mlp.train()
        batch_size = 32
        epochs = 10
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i + batch_size]
                batch_y = y_train_tensor[i:i + batch_size]
                
                optimizer.zero_grad()
                outputs = mlp(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Validation accuracy
            mlp.eval()
            with torch.no_grad():
                val_outputs = mlp(X_val_tensor)
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_accuracy = (val_predictions == y_val_tensor).float().mean()
            
            mlp.train()
            print(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}, Val Acc = {val_accuracy:.4f}")
        
        # Final evaluation
        mlp.eval()
        with torch.no_grad():
            val_outputs = mlp(X_val_tensor)
            val_predictions = torch.argmax(val_outputs, dim=1).numpy()
        
        print("\n📊 Classification Report:")
        print(classification_report(
            y_val, val_predictions, 
            target_names=["BODY", "H1", "H2", "H3"],
            zero_division=0
        ))
        
        # Save trained model
        mlp_path = os.path.join(self.model_dir, "mlp_head.pt")
        torch.save(mlp.state_dict(), mlp_path)
        
        # Save metadata
        metadata = {
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'input_dim': 384 + 7,
            'hidden_dim': 128,
            'num_classes': 4
        }
        
        metadata_path = os.path.join(self.model_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ MLP classifier saved to {mlp_path}")
        print(f"✅ Model metadata saved to {metadata_path}")
        
        return mlp_path
    
    def prepare_all_models(self, proto_labels_dir: str = "proto_labels"):
        """Complete model preparation pipeline."""
        print("🚀 Starting model preparation pipeline...")
        
        # Step 1: Quantize MiniLM
        quantized_path = self.quantize_minilm()
        
        # Step 2: Load prototype corpus
        texts, labels = self.load_prototype_corpus(proto_labels_dir)
        
        # Step 3: Train MLP classifier
        mlp_path = self.train_mlp_classifier(texts, labels, quantized_path)
        
        # Calculate total model size
        total_size = 0
        for root, dirs, files in os.walk(self.model_dir):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        
        total_size_mb = total_size / (1024 * 1024)
        
        print(f"\n✅ Model preparation complete!")
        print(f"📊 Total model size: {total_size_mb:.1f} MB")
        print(f"📁 Models saved in: {self.model_dir}")
        
        if total_size_mb > 200:
            print("⚠️  Warning: Model size exceeds 200MB limit")
        else:
            print("✅ Model size within 200MB limit")


def main():
    parser = argparse.ArgumentParser(description="Prepare models for PDF outline extraction")
    parser.add_argument("--model-dir", default="models", help="Directory to save models")
    parser.add_argument("--proto-labels", default="proto_labels", help="Directory with prototype labels")
    parser.add_argument("--quantize-only", action="store_true", help="Only quantize MiniLM model")
    
    args = parser.parse_args()
    
    preparer = ModelPreparer(args.model_dir)
    
    if args.quantize_only:
        preparer.quantize_minilm()
    else:
        preparer.prepare_all_models(args.proto_labels)


if __name__ == "__main__":
    main()