"""
Utility functions for organizing simulation output files
"""
import os

# Base output directory (relative to project root)
OUTPUT_BASE = "../output"

def get_output_path(category, filename, subfolder=None):
    """
    Get organized output path for generated files
    
    Args:
        category: 'videos', 'graphs', 'analysis', or 'temp_frames'
        filename: name of the file to save
        subfolder: optional subfolder (e.g., 'comms_network')
    
    Returns:
        Full path for the output file
    """
    if subfolder:
        path = os.path.join(OUTPUT_BASE, category, subfolder)
    else:
        path = os.path.join(OUTPUT_BASE, category)
    
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)

def ensure_output_dirs():
    """Ensure all output directories exist"""
    categories = ['videos', 'graphs', 'analysis', 'temp_frames']
    subfolders = ['comms_network']
    
    for category in categories:
        os.makedirs(os.path.join(OUTPUT_BASE, category), exist_ok=True)
        for subfolder in subfolders:
            os.makedirs(os.path.join(OUTPUT_BASE, category, subfolder), exist_ok=True)