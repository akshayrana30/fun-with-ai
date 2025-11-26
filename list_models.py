#!/usr/bin/env python3
"""
Model Discovery and Listing Utility

This script scans the repository for:
1. Python files containing model definitions
2. Saved model files (*.h5, *.pkl, *.pt, *.pth, *.pb)
3. Model architectures and their descriptions
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple


class ModelDiscovery:
    """Discovers and lists ML models in the repository"""
    
    MODEL_FILE_EXTENSIONS = ['.h5', '.pkl', '.pt', '.pth', '.pb', '.joblib', '.sav']
    MODEL_KEYWORDS = ['Sequential', 'Model', 'build_model', 'create_model', 
                      'model =', 'model=', 'Estimator', 'Classifier', 'Regressor',
                      'clf =', 'clf=', '.fit(', 'SVC(', 'GaussianNB(', 'tree.']
    
    def __init__(self, root_dir: str = '.'):
        self.root_dir = Path(root_dir)
        self.models = []
        self.saved_models = []
        self.model_definitions = []
        
    def scan_directory(self) -> None:
        """Scan the directory for models"""
        print(f"Scanning directory: {self.root_dir.absolute()}\n")
        print("=" * 80)
        
        # Find saved model files
        self._find_saved_models()
        
        # Find model definitions in Python files
        self._find_model_definitions()
        
    def _find_saved_models(self) -> None:
        """Find saved model files"""
        print("\n📦 SAVED MODEL FILES")
        print("-" * 80)
        
        found = False
        for ext in self.MODEL_FILE_EXTENSIONS:
            for model_file in self.root_dir.rglob(f"*{ext}"):
                if '.git' not in str(model_file):
                    self.saved_models.append(model_file)
                    found = True
                    size = model_file.stat().st_size if model_file.exists() else 0
                    size_mb = size / (1024 * 1024)
                    rel_path = model_file.relative_to(self.root_dir)
                    print(f"  ✓ {rel_path}")
                    print(f"    Size: {size_mb:.2f} MB")
        
        if not found:
            print("  No saved model files found in the repository.")
            
    def _find_model_definitions(self) -> None:
        """Find model definitions in Python files"""
        print("\n🔍 MODEL IMPLEMENTATIONS")
        print("-" * 80)
        
        python_files = list(self.root_dir.rglob("*.py"))
        
        for py_file in python_files:
            if '.git' not in str(py_file):
                self._analyze_python_file(py_file)
                
    def _analyze_python_file(self, file_path: Path) -> None:
        """Analyze a Python file for model definitions"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if file contains model-related code
            has_model = any(keyword in content for keyword in self.MODEL_KEYWORDS)
            
            if has_model:
                rel_path = file_path.relative_to(self.root_dir)
                print(f"\n📄 {rel_path}")
                
                # Extract model architecture information
                model_info = self._extract_model_info(content, file_path)
                
                if model_info:
                    for info in model_info:
                        print(f"  • {info}")
                        
                self.model_definitions.append(str(rel_path))
                
        except Exception as e:
            # Silently skip files that can't be read
            pass
            
    def _extract_model_info(self, content: str, file_path: Path) -> List[str]:
        """Extract model information from file content"""
        info = []
        frameworks = []
        
        # Check for Keras Sequential models
        if 'Sequential' in content:
            info.append("Type: Keras Sequential Model")
            
        # Check for TensorFlow
        if 'tensorflow' in content or 'import tf' in content:
            frameworks.append("TensorFlow")
            
        # Check for PyTorch
        if 'torch' in content:
            frameworks.append("PyTorch")
            
        # Check for scikit-learn
        if 'sklearn' in content:
            frameworks.append("scikit-learn")
            
        # Add frameworks to info
        if frameworks:
            info.append(f"Framework: {', '.join(frameworks)}")
            
        # Extract function definitions related to models
        func_pattern = r'def\s+(build_model|create_model|train_model|load_model)\s*\('
        functions = re.findall(func_pattern, content)
        if functions:
            info.append(f"Functions: {', '.join(set(functions))}")
            
        # Detect specific model types
        if 'SVM' in content or 'svm' in content:
            info.append("Algorithm: Support Vector Machine (SVM)")
        if 'GaussianNB' in content or 'naive_bayes' in content:
            info.append("Algorithm: Gaussian Naive Bayes")
        if 'tree' in content and 'sklearn' in content:
            info.append("Algorithm: Decision Tree")
        if 'Conv2D' in content or 'Convolution2D' in content:
            info.append("Architecture: Convolutional Neural Network (CNN)")
        if 'LSTM' in content or 'GRU' in content:
            info.append("Architecture: Recurrent Neural Network (RNN)")
            
        # Extract model architecture comments/docstrings
        arch_pattern = r'(?:"""|\'\'\')(.*?model.*?)(?:"""|\'\'\')'
        architectures = re.findall(arch_pattern, content, re.IGNORECASE | re.DOTALL)
        for arch in architectures[:1]:  # Only show first one
            arch_clean = ' '.join(arch.split()[:20])  # First 20 words
            if len(arch_clean) > 10:
                info.append(f"Description: {arch_clean}...")
                
        # Count layers (rough estimate)
        layer_count = content.count('.add(')
        if layer_count > 0:
            info.append(f"Approximate layers: {layer_count}")
            
        return info
        
    def print_summary(self) -> None:
        """Print a summary of findings"""
        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)
        print(f"Saved model files found: {len(self.saved_models)}")
        print(f"Python files with model definitions: {len(self.model_definitions)}")
        
        if self.model_definitions:
            print("\nModel implementation files:")
            for model_def in sorted(self.model_definitions):
                print(f"  • {model_def}")


def main():
    """Main function"""
    print("🤖 Model Discovery Utility for fun-with-ai Repository")
    print("=" * 80)
    
    # Get the repository root directory
    script_dir = Path(__file__).parent
    
    # Create discovery instance and scan
    discovery = ModelDiscovery(script_dir)
    discovery.scan_directory()
    discovery.print_summary()
    
    print("\n" + "=" * 80)
    print("✅ Scan complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
