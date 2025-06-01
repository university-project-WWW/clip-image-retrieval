"""
Create submission file for image retrieval competition.
"""

import json
from datetime import datetime
from pathlib import Path

from config import JSON_DIR

def create_submission(query_filenames, gallery_filenames, topk_indices, output_path, k=10):
    """
    Create submission file in the required JSON format.
    
    Format:
    [
        {
            "filename": "query_image.jpg",
            "samples": ["gallery_image1.jpg", "gallery_image2.jpg", ...]
        },
        ...
    ]
    """
    # Create backup directory if it doesn't exist
    JSON_DIR.mkdir(exist_ok=True)
    
    # Extract the original filename without path
    original_filename = Path(output_path).name
    base_name = original_filename.split('.')[0]
    extension = original_filename.split('.')[-1]
    
    # Generate timestamp for the backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup_file = JSON_DIR / f"{base_name}_{timestamp}.{extension}"
    
    # Create submission data
    submission = [
        {
            "filename": query_filenames[i],
            "samples": [gallery_filenames[j] for j in topk_indices[i][:k]]
        }
        for i in range(len(query_filenames))
    ]
    
    # Save the primary submission with original name
    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)
    
    # Save the backup with timestamp
    with open(backup_file, "w") as f:
        json.dump(submission, f, indent=2)
    
    print(f" {original_filename} saved to {output_path}. "
          f"Backup saved to {backup_file}. "
          f"Contains {len(submission)} entries × top-{k}.")
    
    return output_path

def create_submission_with_tta(query_filenames, gallery_filenames, topk_indices, 
                            output_path, k=10, technique="tta"):
    """
    Create submission file with advanced techniques like TTA or re-ranking.
    Adds technique name to backup filename.
    """
    # Create backup directory if it doesn't exist
    JSON_DIR.mkdir(exist_ok=True)
    
    # Extract the original filename without path
    original_filename = Path(output_path).name
    base_name = original_filename.split('.')[0]
    extension = original_filename.split('.')[-1]
    
    # Generate timestamp for the backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup_file = JSON_DIR / f"{base_name}_{technique}_{timestamp}.{extension}"
    
    # Create submission data
    submission = [
        {
            "filename": query_filenames[i],
            "samples": [gallery_filenames[j] for j in topk_indices[i][:k]]
        }
        for i in range(len(query_filenames))
    ]
    
    # Save the primary submission with original name
    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)
    
    # Save the backup with technique name and timestamp
    with open(backup_file, "w") as f:
        json.dump(submission, f, indent=2)
    
    print(f" {technique.upper()} submission saved to {output_path}. "
          f"Backup saved to {backup_file}. "
          f"Contains {len(submission)} entries × top-{k}.")
    
    return output_path
