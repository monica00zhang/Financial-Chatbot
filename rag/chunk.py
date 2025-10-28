"""
Buffett Letter to Shareholder Parser
Extract and organize information from Berkshire Hathaway annual letters
"""

import re
import json
from datetime import datetime
from PyPDF2 import PdfReader
from itertools import product


FINANCIAL_TERMS = {
    'Gross Margin': r'gross margin',
    'Profit Margin': r'profit margin',
    'Earnings Per Share Growth': r'earnings per share growth',
    'SG & A Expense Margin': r'sg\s*&\s*a expense margin|sg&a|selling, general and administrative expense margin',
    'Interest Expense Margin': r'interest expense margin',
    'Depreciation Margin': r'depreciation margin',
    'R & D Expense Margin': r'r\s*&\s*d expense margin|research and development expense margin',
    'Cash > Debt': r'cash\s*>\s*debt',
    'Adjusted Debt to Equity': r'adjusted debt to equity',
    'CapEx Margin': r'capex margin'
}

COMPANY_PATTERN = r'([A-Z][a-zA-Z\s]+ (?:Inc\.|Corp\.|Ltd\.|LLC|Co\.|Company))'
DATE_PATTERN = r'([A-Za-z]+ \d{1,2}, ?\d{4})'

companies = {
    "insurance": [
        'National Indemnity Company',
        'National Fire and Marine Insurance Company',
        'Cypress Insurance Company',
        'Cornhusker Casualty Company',
        'Lakeland Fire and Casualty Company',
        'Texas United Insurance Company',
        'Home and Automobile Insurance Company',
        'Kansas Fire and Casualty Company',
        'Central Fire and Casualty Company',
        'Government Employees Insurance                   Company',
        'Government Employees Insurance Company',
        'SAFECO Corp.',
        'Chubb Corp.',
        'Columbia Insurance Company',
        'Medical Protective Company',
        'White Mountains Insurance Group Ltd.',
        'Berkshire Hathaway Assurance Company',
        'Financial Insurance Company',
        'General Reinsurance Company',
        'GEICO Corp.',
        'Government Employees Insurance Co.',
    ],
    "banking_financial": [
        'Illinois National Bank and Trust Company',
        'Rockford Bancorp Inc.',
        'Wesco Financial Corp.',
        'Leucadia National Corp.',
        'PNC Bank Corp.',
        'Bank of America Corp.',
        'The Bank of New York Mellon Corp.',
        'Resolution Trust Corp.',
        'American Express Company',
        'First Empire State Corp.',
        'Buffett Partnership Ltd.',
        'Diversified Retailing Company',
        'Fetzer Company',
    ],
    "media": [
        'The Washington Post Company',
        'Buffalo News and our shareholdings in The Washington Post Company',
        'Gannett Co.',
        'The Times Mirror Company',
        'E W Woolworth Company',
    ],
    "retail_consumer": [
        'Woolworth Company',
        'VF Corp.',
        'Russell Corp.',
        'Champion International Corp.',
        'Melville Corp.',
        'Golden West Financial and Price Co.',
        'Brown Company',
        'Lowell Shoe Company',
        'The Gillette Company',
        'The Gillette  Company',
        'Gillette Co.',
        'Best Co.',
        'Girls Inc.',
        'Kraft Foods Inc.',
        'Cola Company',
        'Cola Co.',
        'CEO of Salomon Inc.'
    ],
    "energy_utilities": [
        'Exxon Mobil Corp.',
        'Exxon Corp.',
        'CEO of Exxon Corp.',
        'PetroChina Company',
        'General Dynamics Corp.',
    ],
    "new_energy": [
        'BYD Company',
        'BYD Company Ltd.'
    ],
    "technology": [
        'Apple Inc.',
        'International Business Machines Corp.',
        'Visa Inc.',
    ],
    "industrial_manufacturing": [
        'Precision Castparts Corp.',
        'Cliffs Iron Company',
        'Mather International Inc.',
        'General Motors Company',
        'Johns Manville Corp.',
        'Aluminum Company',
        'USG Corp.',
    ],
    "transportation": [
        'Delta Airlines Inc.',
        'United Continental Holdings Inc.',
        'Southwest Airlines Co.',
    ],
    "healthcare": [
        'HCA Inc.',
        'DaVita HealthCare Partners Inc.',
    ],
    "others": [
        'Chemical Corp.',
        'Wachovia Corp.',
        'General Re Corp.',
        'Gamble Company',
        'The Walt Disney Company',
        'Chairman of General Reinsurance Company',
        'Chairman of the Investment Company',
        'CEO of The Walt Disney Company',
        'Portrait of a Disciplined Underwriter  National Indemnity Company',
    ]
}


# Paragraph split : Buffett's Letter to shareholder

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def clean_paragraphs(letter_text):
    paragraphs = [p for p in letter_text.split('\n \n    ') if len(p) > 0]
    paragraphs = [y for x in paragraphs for y in x.split("\n \n")]
    paragraphs = [y for x in paragraphs for y in x.split(".\n")]
    paragraphs = [x.replace('\n', " ").replace('   ', " ").replace('  ', " ") for x in paragraphs]
    return paragraphs


# ==================       Extract mentioned Keywords:  date, company names, financial terms     ======================
def extract_date(letter_text):
    date_match = re.findall(DATE_PATTERN, letter_text[-100:])
    if date_match:
        date_str = date_match[0].replace(',', ', ')
        dt = datetime.strptime(date_str, '%B %d, %Y')
        return dt.strftime('%Y%m%d')
    return None


def extract_companies(letter_text):
    company_mentions = re.findall(COMPANY_PATTERN, letter_text)
    return list(set(company_mentions))


def extract_financial_terms(paragraphs):
    financial_term_indices = {}
    for term, pattern in FINANCIAL_TERMS.items():
        matched_paragraphs = []
        for para_idx, para in enumerate(paragraphs):
            if re.search(pattern, para, re.IGNORECASE):
                matched_paragraphs.append(para_idx)
        if matched_paragraphs:
            financial_term_indices[term] = matched_paragraphs
    return financial_term_indices


# ==================       Get index of  mentioned Keywords:  date, company names, financial terms     ======================
def parse_letters(pdf_path):
    """
    Main parser: extract and structure all letters
    
    Args:
        pdf_path: path to PDF file
    
    Returns:
        tuple of (meta_letter, letter_detail)
    """
    # Extract text
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Paragraph: Split into individual letters
    letters = extracted_text.split("BERKSHIRE HATHAWAY INC. \n")[1:]
    
    meta_letter = {}
    letter_detail = {}
    
    for i, letter in enumerate(letters, start=1):
        # Clean paragraphs
        paragraphs = clean_paragraphs(letter)
        
        # Store meta information
        meta_letter[i] = {
            "letter_index": i,
            "paragraphs": paragraphs
        }
        
        # Extract detailed information
        date = extract_date(letter)
        if not date:
            print(f"Warning: No date found in letter {i}")
        
        companies = extract_companies(letter)
        financial_terms = extract_financial_terms(paragraphs)
        
        letter_detail[i] = {
            "date": date,
            "letter_index": i,
            "company_mentions": companies,
            "financial_terms": financial_terms
        }
    
    return meta_letter, letter_detail


def convert_companies_with_paragraphs(companies, letter_detail, meta_letter):
    """
    Map companies to their appearances in letters with paragraph indices
    
    Args:
        companies: dict of {sector: [company_list]}
        letter_detail: letter detail dictionary
        meta_letter: meta letter dictionary
    
    Returns:
        dict of {sector: {company: {letter_index: [para_indices]}}}
    """
    result = {}
    
    for sector, company_list in companies.items():
        sector_dict = {}
        for company in company_list:
            appearances = {}
            for letter_index, detail in letter_detail.items():
                mentions = detail.get("company_mentions", [])
                if any(company in mention for mention in mentions):
                    paragraphs = meta_letter.get(letter_index, {}).get("paragraphs", [])
                    matching_indices = []
                    for idx, paragraph in enumerate(paragraphs):
                        if company in paragraph:
                            matching_indices.append(idx)
                    if matching_indices:
                        appearances[letter_index] = matching_indices
            if appearances:
                sector_dict[company] = appearances
        if sector_dict:
            result[sector] = sector_dict
    
    return result


def extract_financial_terms_from_letters(term_list, letter_detail):
    """
    Extract all occurrences of financial terms across letters
    
    Args:
        term_list: list of financial term names
        letter_detail: letter detail dictionary
    
    Returns:
        dict of {term: [{letter_index: para_index}, ...]}
    """
    result = {}
    
    for term in term_list:
        term_entries = []
        for letter_index, detail in letter_detail.items():
            paragraph_indices = detail.get("financial_terms", {}).get(term)
            if paragraph_indices:
                for para_index in paragraph_indices:
                    term_entries.append({letter_index: para_index})
        if term_entries:
            result[term] = term_entries
    
    return result


def save_to_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)





def main():
    # Parse letters
    print("Parsing Buffett letters...")
    pdf_path = "Dataset/All-Berkshire-Hathaway-Letters.pdf"
    meta_letter, letter_detail = parse_letters(pdf_path)
    print(f"Parsed {len(meta_letter)} letters")
    
    
    # Convert company appearances
    print("\nExtracting company mentions...")
    company_paragraphs = convert_companies_with_paragraphs(companies, letter_detail, meta_letter)
    
    # Extract financial term occurrences
    print("Extracting financial terms...")
    terms = list(FINANCIAL_TERMS.keys())
    financial_paragraphs = extract_financial_terms_from_letters(terms, letter_detail)
    
    # Save results
    print("\nSaving results...")
    save_to_json(meta_letter, "Dataset/meta_letter.json")
    save_to_json(company_paragraphs, "Dataset/company_exa.json")
    save_to_json(financial_paragraphs, "Dataset/indicator_exa.json")
    
    print("\n Chunk Processing complete!")


if __name__ == "__main__":
    main()