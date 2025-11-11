#!/usr/bin/env python3
"""
Script to remove comments from all Python files in the project
"""

import os
import re
from pathlib import Path

def remove_comments_from_file(file_path):
    """Remove comments from a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove single-line comments (but preserve # in strings and URLs)
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                cleaned_lines.append(line)
                continue
            
            # Find the first occurrence of # that's not in a string
            in_string = False
            string_char = None
            comment_pos = -1
            
            for i, char in enumerate(line):
                if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None
                elif char == '#' and not in_string:
                    comment_pos = i
                    break
            
            if comment_pos >= 0:
                # Keep everything before the comment
                cleaned_line = line[:comment_pos].rstrip()
                # Only add the line if it has content before the comment
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)
                # If no content before comment, add empty line
                else:
                    cleaned_lines.append('')
            else:
                cleaned_lines.append(line)
        
        # Join lines back
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Remove docstrings (triple quotes) - more aggressive approach
        # Remove """...""" docstrings
        cleaned_content = re.sub(r'^\s*""".*?"""\s*$', '', cleaned_content, flags=re.MULTILINE | re.DOTALL)
        # Remove '''...''' docstrings  
        cleaned_content = re.sub(r"^\s*'''.*?'''\s*$", '', cleaned_content, flags=re.MULTILINE | re.DOTALL)
        
        # Remove inline docstrings that might have been missed
        cleaned_content = re.sub(r'""".*?"""', '', cleaned_content, flags=re.DOTALL)
        cleaned_content = re.sub(r"'''.*?'''", '', cleaned_content, flags=re.DOTALL)
        
        # Clean up multiple empty lines and trailing whitespace
        cleaned_content = re.sub(r'[ \t]+$', '', cleaned_content, flags=re.MULTILINE)
        cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
        
        # Remove leading/trailing whitespace
        cleaned_content = cleaned_content.strip()
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print(f"✅ Cleaned: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Main function to process all Python files"""
    project_root = Path("d:\\dheer@j\\credit-card-fraud-detection")
    python_files = [
        "demo_complete.py",
        "demo_dashboard_improved.py", 
        "demo_dashboard.py",
        "demo_system.py",
        "scripts\\deploy.py",
        "scripts\\evaluate_model.py",
        "scripts\\predict.py",
        "scripts\\train_model.py",
        "src\\api\\__init__.py",
        "src\\api\\main.py",
        "src\\data\\preprocessing.py",
        "src\\evaluation\\__init__.py",
        "src\\evaluation\\metrics.py",
        "src\\models\\__init__.py",
        "src\\models\\fraud_models.py",
        "src\\utils\\__init__.py",
        "src\\utils\\config.py",
        "src\\utils\\error_handling.py",
        "tests\\conftest.py",
        "tests\\integration\\test_integration.py",
        "tests\\performance\\test_performance.py",
        "tests\\unit\\test_fraud_detection.py"
    ]
    
    success_count = 0
    total_count = len(python_files)
    
    print(f"🚀 Starting comment removal for {total_count} Python files...")
    print("=" * 60)
    
    for file_name in python_files:
        file_path = project_root / file_name
        if file_path.exists():
            if remove_comments_from_file(file_path):
                success_count += 1
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print("=" * 60)
    print(f"📊 Summary: {success_count}/{total_count} files processed successfully")
    
    if success_count == total_count:
        print("🎉 All files cleaned successfully!")
    else:
        print("⚠️  Some files had issues. Check the output above.")

if __name__ == "__main__":
    main()