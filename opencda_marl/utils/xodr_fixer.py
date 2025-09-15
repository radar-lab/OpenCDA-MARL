'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-29 09:35:35
FilePath     : /OpenCDA-MARL/opencda_marl/utils/xodr_fixer.py
Description  : Utility to fix common issues in XODR files for CARLA compatibility.

This utility handles:
1. Adding missing georeference tags
2. Ensuring proper CDATA format for georeference content
3. Creating backups before modifying files

Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
import os
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil
import argparse


def add_georeference_to_xodr(xodr_path: str, output_path: str = None, use_cdata: bool = True):
    """
    Add a georeference tag to an XODR file if it's missing.
    
    The georeference uses a standard projection string (UTM Zone 17N as default).
    This eliminates the CARLA warning about missing georeference.
    
    Args:
        xodr_path: Path to the input XODR file
        output_path: Path for the output file (if None, overwrites input)
        use_cdata: Whether to use CDATA format for georeference content
    """
    
    # Parse the XODR file
    tree = ET.parse(xodr_path)
    root = tree.getroot()
    
    # Check if geoReference already exists (either as child of header or root)
    header = root.find('header')
    existing_georef = None
    
    if header is not None:
        existing_georef = header.find('geoReference')
    
    if existing_georef is None:
        # Also check if it's incorrectly placed as a child of root
        existing_georef = root.find('geoReference')
        if existing_georef is not None:
            # Remove it from wrong location
            root.remove(existing_georef)
            print("Removed incorrectly placed geoReference from root")
            existing_georef = None
    
    if existing_georef is None:
        # Create a georeference element with a standard projection
        # Using lat/lon format that CARLA expects
        # This is a simple lat/lon projection centered at a default location
        georef_text = "+lat_0=4.9000000000000000e+1 +lon_0=8.0000000000000000e+0"
        
        # Find the header element
        header = root.find('header')
        
        if header is not None:
            # CARLA expects geoReference as a child of header, not sibling
            # Create the geoReference element
            georef = ET.SubElement(header, 'geoReference')
            georef.text = georef_text
            
            print("Added geoReference to XODR file")
        else:
            print("Warning: Could not find header element in XODR")
            return False
    else:
        print("geoReference already exists in XODR file")
    
    # Write the modified XML
    if output_path is None:
        output_path = xodr_path
        
    # If using CDATA format, use string replacement method for better control
    if use_cdata:
        return _write_with_cdata(root, output_path)
    else:
        return _write_standard_xml(root, output_path)


def _write_with_cdata(root, output_path: str) -> bool:
    """
    Write XML with proper CDATA format for geoReference.
    This uses string replacement to ensure CDATA is properly formatted.
    """
    # Convert to string first
    xml_str = ET.tostring(root, encoding='unicode')
    
    # Use regex to find and replace geoReference content with CDATA
    pattern = r'<geoReference>([^<]+)</geoReference>'
    match = re.search(pattern, xml_str)
    
    if match:
        old_text = match.group(0)
        geo_content = match.group(1).strip()
        new_text = f'<geoReference><![CDATA[{geo_content}]]></geoReference>'
        xml_str = xml_str.replace(old_text, new_text)
        print("Applied CDATA format to geoReference")
    
    # Pretty format the XML
    try:
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="    ")
        
        # Remove extra blank lines
        lines = pretty_xml.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        pretty_xml = '\n'.join(non_empty_lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        
        print(f"Fixed XODR saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error writing CDATA format: {e}")
        return False


def _write_standard_xml(root, output_path: str) -> bool:
    """Write XML in standard format without CDATA."""
    try:
        # Pretty print the XML
        xml_str = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="    ")
        
        # Remove extra blank lines
        lines = pretty_xml.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        pretty_xml = '\n'.join(non_empty_lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        
        print(f"Fixed XODR saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error writing standard XML: {e}")
        return False


def fix_georeference_with_cdata(xodr_path: str) -> bool:
    """
    Fix existing geoReference to use proper CDATA format.
    This function specifically handles the CDATA format requirement.
    
    Args:
        xodr_path: Path to the XODR file to fix
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the file as text
        with open(xodr_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if CDATA is already present
        if '<![CDATA[' in content and ']]>' in content:
            print("CDATA format already present in geoReference")
            return True
        
        # Try to find and replace geoReference with CDATA format
        pattern = r'<geoReference>([^<]+)</geoReference>'
        match = re.search(pattern, content)
        
        if match:
            old_text = match.group(0)
            geo_content = match.group(1).strip()
            new_text = f'<geoReference><![CDATA[{geo_content}]]></geoReference>'
            content = content.replace(old_text, new_text)
            print("Replaced geoReference with CDATA format")
            
            # Write back
            with open(xodr_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Fixed XODR saved to: {xodr_path}")
            return True
        else:
            print("Could not find geoReference to fix")
            return False
            
    except Exception as e:
        print(f"Error fixing geoReference with CDATA: {e}")
        return False


def fix_marl_maps(use_cdata: bool = True):
    """
    Fix all MARL maps by adding georeference if needed.
    
    This function:
    1. Creates backups of original files
    2. Adds missing georeference tags
    3. Ensures proper CDATA format for CARLA compatibility
    
    Args:
        use_cdata: Whether to use CDATA format (recommended for CARLA)
    """
    maps_dir = os.path.join(
        os.path.dirname(__file__),
        "../../assets/maps"
    )
    
    print(f"Checking MARL maps in: {maps_dir}")
    
    # Fix intersection.xodr
    intersection_xodr = os.path.join(maps_dir, "intersection.xodr")
    
    if os.path.exists(intersection_xodr):
        print("Processing intersection.xodr...")
        
        # Create backup first
        backup_path = intersection_xodr + ".backup"
        if not os.path.exists(backup_path):
            shutil.copy2(intersection_xodr, backup_path)
            print(f"Created backup: {backup_path}")
        
        # First, ensure georeference exists
        success = add_georeference_to_xodr(intersection_xodr, use_cdata=use_cdata)
        
        if success and use_cdata:
            # Then ensure CDATA format is applied
            fix_georeference_with_cdata(intersection_xodr)
        
        print("✓ intersection.xodr processing complete")
    else:
        print(f"✗ intersection.xodr not found at: {intersection_xodr}")
    
    print("MARL maps fixing complete!")


def validate_fixed_maps():
    """
    Validate that MARL maps have been properly fixed.
    
    Returns:
        bool: True if all maps are valid, False otherwise
    """
    maps_dir = os.path.join(
        os.path.dirname(__file__),
        "../../assets/maps"
    )
    
    intersection_xodr = os.path.join(maps_dir, "intersection.xodr")
    
    if not os.path.exists(intersection_xodr):
        print("✗ intersection.xodr not found")
        return False
    
    try:
        # Read and check for georeference with CDATA
        with open(intersection_xodr, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for georeference tag
        if '<geoReference>' not in content:
            print("✗ geoReference tag missing")
            return False
        
        # Check for CDATA format
        if '<![CDATA[' in content and ']]>' in content:
            print("✓ geoReference with CDATA format found")
            return True
        else:
            print("⚠ geoReference found but no CDATA format")
            return True  # Still valid, just not optimal
            
    except Exception as e:
        print(f"✗ Error validating maps: {e}")
        return False


if __name__ == "__main__":
    print("OpenCDA-MARL Map Fixer")
    print("=" * 40)
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cdata", action="store_true", help="Use CDATA format for georeference")
    args = parser.parse_args()
    
    # Fix the maps
    fix_marl_maps(use_cdata=args.use_cdata)
    
    print("\nValidating fixed maps...")
    if validate_fixed_maps():
        print("✓ All maps are properly formatted!")
    else:
        print("✗ Some issues found with map formatting")