import pandas as pd
import re
import os
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def try_import_tabula():
    """Try to import tabula-py, return None if not available"""
    try:
        import tabula
        return tabula
    except ImportError:
        logger.warning("tabula-py not available. Install with: pip install tabula-py")
        return None

def try_import_camelot():
    """Try to import camelot-py, return None if not available"""
    try:
        import camelot
        return camelot
    except ImportError:
        logger.warning("camelot-py not available. Install with: pip install camelot-py[cv]")
        return None

def extract_table_name_from_text(text):
    """
    Extract table name from text that contains table headers like 'Table 1: List of LNG thermal power plants'
    """
    if not text:
        return "Unknown_Table"
    
    # Look for patterns like "Table X: Description" or "Table X - Description"
    table_patterns = [
        r'Table\s+(\d+):\s*(.+)',
        r'Table\s+(\d+)\s*-\s*(.+)',
        r'Table\s+(\d+)\s+(.+)',
        r'Bảng\s+(\d+):\s*(.+)',  # Vietnamese "Bảng" = Table
        r'Bảng\s+(\d+)\s*-\s*(.+)'
    ]
    
    for pattern in table_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            table_num = match.group(1)
            table_desc = match.group(2).strip()
            # Clean the description for use as worksheet name
            clean_desc = re.sub(r'[^\w\s-]', '', table_desc)
            clean_desc = re.sub(r'\s+', '_', clean_desc)
            return f"Table_{table_num}_{clean_desc}"
    
    return "Unknown_Table"

def clean_table_data(df):
    """
    Clean and process the extracted table data
    """
    if df is None or df.empty:
        return df
    
    # Remove completely empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Clean column names
    if not df.columns.empty:
        df.columns = [str(col).strip() if pd.notna(col) else f'Column_{i}' 
                     for i, col in enumerate(df.columns)]
    
    # Clean cell values
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(['nan', 'None', ''], pd.NA)
    
    return df

def extract_tables_with_tabula(pdf_path):
    """Extract tables using tabula-py"""
    tabula = try_import_tabula()
    if tabula is None:
        return []
    
    try:
        logger.info("Extracting tables with tabula-py...")
        tables = tabula.read_pdf(
            pdf_path,
            pages='all',
            multiple_tables=True,
            guess=False,
            lattice=True,
            stream=True,
            pandas_options={'header': None}
        )
        return tables
    except Exception as e:
        logger.error(f"Error with tabula-py: {e}")
        return []

def extract_tables_with_camelot(pdf_path):
    """Extract tables using camelot-py"""
    camelot = try_import_camelot()
    if camelot is None:
        return []
    
    try:
        logger.info("Extracting tables with camelot-py...")
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        return [table.df for table in tables]
    except Exception as e:
        logger.error(f"Error with camelot-py: {e}")
        return []

def extract_tables_from_pdf(pdf_path, output_path):
    """
    Extract tables from PDF using multiple methods and save to Excel with separate worksheets
    """
    logger.info(f"Starting extraction from: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Try different extraction methods
    all_tables = []
    
    # Method 1: Try tabula-py
    tabula_tables = extract_tables_with_tabula(pdf_path)
    if tabula_tables:
        all_tables.extend(tabula_tables)
        logger.info(f"Extracted {len(tabula_tables)} tables with tabula-py")
    
    # Method 2: Try camelot-py if tabula didn't work well
    if not all_tables:
        camelot_tables = extract_tables_with_camelot(pdf_path)
        if camelot_tables:
            all_tables.extend(camelot_tables)
            logger.info(f"Extracted {len(camelot_tables)} tables with camelot-py")
    
    if not all_tables:
        raise Exception("No tables could be extracted from the PDF. Please check if the PDF contains tables.")
    
    logger.info(f"Total tables found: {len(all_tables)}")
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        table_count = 0
        successful_tables = 0
        
        for i, table in enumerate(all_tables):
            if table is None or table.empty:
                continue
            
            # Clean the table data
            cleaned_table = clean_table_data(table)
            
            if cleaned_table is None or cleaned_table.empty:
                continue
            
            # Try to extract table name from the first few rows
            table_name = "Unknown_Table"
            
            # Look for table headers in the first few rows
            for row_idx in range(min(5, len(cleaned_table))):
                row_text = ' '.join(str(cell) for cell in cleaned_table.iloc[row_idx] if pd.notna(cell))
                extracted_name = extract_table_name_from_text(row_text)
                if extracted_name != "Unknown_Table":
                    table_name = extracted_name
                    break
            
            # If no table name found, use a generic name
            if table_name == "Unknown_Table":
                table_count += 1
                table_name = f"Table_{table_count}"
            
            # Ensure worksheet name is valid (Excel has restrictions)
            worksheet_name = table_name[:31]  # Excel worksheet names limited to 31 characters
            worksheet_name = re.sub(r'[\\/*?:\[\]]', '_', worksheet_name)
            
            # Handle duplicate worksheet names
            original_name = worksheet_name
            counter = 1
            while worksheet_name in [sheet.title for sheet in writer.book.worksheets]:
                worksheet_name = f"{original_name}_{counter}"[:31]
                counter += 1
            
            logger.info(f"Processing table {i+1}: {worksheet_name}")
            
            # Write table to Excel
            try:
                cleaned_table.to_excel(writer, sheet_name=worksheet_name, index=False)
                logger.info(f"Successfully wrote table to worksheet: {worksheet_name}")
                successful_tables += 1
            except Exception as e:
                logger.error(f"Error writing table {i+1} to Excel: {e}")
                # Try with a simpler name
                simple_name = f"Table_{i+1}"
                cleaned_table.to_excel(writer, sheet_name=simple_name, index=False)
                logger.info(f"Wrote table with simple name: {simple_name}")
                successful_tables += 1
    
    logger.info(f"Extraction complete. Successfully processed {successful_tables} tables.")
    logger.info(f"Output saved to: {output_path}")
    return output_path

def main():
    """
    Main function to run the PDF table extraction
    """
    # Define file paths
    pdf_path = r"C:\Users\SamClissold\MCCF_Git_Repo\MCCF\PDP8\data\PDP8 power projects (english translation).pdf"
    output_path = r"C:\Users\SamClissold\MCCF_Git_Repo\MCCF\PDP8\data\PDP8_power_projects_tables.xlsx"
    
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract tables from PDF
        result_path = extract_tables_from_pdf(pdf_path, output_path)
        
        print(f"\n✅ Successfully extracted tables from PDF!")
        print(f"📁 Output file: {result_path}")
        print(f"📊 Check the Excel file for extracted tables in separate worksheets.")
        print(f"\n💡 If you need better table extraction, consider installing:")
        print(f"   - tabula-py: pip install tabula-py")
        print(f"   - camelot-py: pip install camelot-py[cv]")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"❌ Error: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 