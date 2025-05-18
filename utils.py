import os
import urllib.parse

def format_clickable_path(path):
    absolute_path = os.path.abspath(path)
    encoded_path = urllib.parse.quote(absolute_path)
    return f"file://{encoded_path}"