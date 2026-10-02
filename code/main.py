#!/usr/bin/env python3
import os
import sys
import json
import argparse
import requests
import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. LCSC Search Tool (internal API call)
# ---------------------------------------------------------------------------
def search_lcsc(query: str, limit: int = 5):
    """Search the LCSC catalog through its internal API."""
    url = "https://www.lcsc.com/api/products/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Accept": "application/json"
    }
    params = {"keyword": query, "curr": "USD"}
    
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            products = data.get("result", {}).get("productList", [])[:limit]
            results = []
            for p in products:
                results.append({
                    "lcsc_code": p.get("productCode"),       # e.g. C2368
                    "mfr_part": p.get("lightProductModel"),  # e.g. CL10B104KO8NNNC
                    "brand": p.get("lightBrandNameEn"),      # e.g. Samsung
                    "package": p.get("encapStandard"),       # e.g. 0603
                    "stock": p.get("stockNumber"),           # Stock quantity
                    "description": p.get("productIntroEn")
                })
            return results
    except Exception as e:
        return {"error": str(e)}
    return []

# Tool definition for the DeepSeek API
TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_lcsc",
        "description": "Search the LCSC parts catalog by keyword or part number.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. '0603 100nf 50v' or 'STM32F401RCT6'"
                }
            },
            "required": ["query"]
        }
    }
}]

# ---------------------------------------------------------------------------
# 2. Agent logic
# ---------------------------------------------------------------------------
def find_lcsc_part(client, reference, value, footprint, mpn=""):
    prompt = f"""
    Your task is to select the most suitable LCSC part code (C-code) for this KiCad BOM item:
    - Reference: {reference}
    - Value/Parameters: {value}
    - Footprint/Package: {footprint}
    - MFR Part Number: {mpn}

    Instructions:
    1. Use the `search_lcsc` function to search.
    2. If you find an in-stock part with a matching package and value, select the best match.
    3. Prefer standard parts with good stock availability.
    4. Reply ONLY with a valid JSON object; do not include any other text.
    Format: {{"lcsc_code": "Cxxxxx", "mfr_part": "...", "confidence": "high/medium/low", "note": "..."}}
    """

    messages = [{"role": "user", "content": prompt}]
    
    # Round 1: DeepSeek formulates the search query
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )
    
    msg = response.choices[0].message
    
    # If the LLM requests a search
    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            search_results = search_lcsc(args["query"])
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(search_results)
            })
        
        # Round 2: DeepSeek selects the best match from the results
        final_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        return final_response.choices[0].message.content

    return msg.content

# ---------------------------------------------------------------------------
# 3. Main program and CLI handling
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Add LCSC C-codes to a KiCad BOM using the DeepSeek LLM.")
    parser.add_argument("file", nargs="?", help="Path to the exported KiCad CSV BOM file")
    args = parser.parse_args()

    # If no file was provided on the command line, prompt for one interactively
    file_path = args.file
    if not file_path:
        file_path = input("Drag the KiCad BOM (.csv) file here or enter its path: ").strip()
        # Remove surrounding quotes from paths pasted on Windows
        file_path = file_path.strip("'\"")

    if not os.path.exists(file_path):
        print(f"[-] Error: File not found: {file_path}")
        sys.exit(1)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        api_key = input("DEEPSEEK_API_KEY was not found. Enter your API key: ").strip()

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print(f"\n[*] Reading BOM: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[-] Error reading CSV: {e}")
        sys.exit(1)

    # Detect supported column-name variants
    ref_col = next((c for c in df.columns if c.lower() in ['reference', 'designator', 'refs']), None)
    val_col = next((c for c in df.columns if c.lower() in ['value', 'val']), None)
    foot_col = next((c for c in df.columns if c.lower() in ['footprint', 'package']), None)
    mpn_col = next((c for c in df.columns if c.lower() in ['mpn', 'mfr_part', 'manufacturer part number']), None)

    if not ref_col or not val_col:
        print("[-] Error: The CSV file must contain a 'Reference' and a 'Value' column.")
        sys.exit(1)

    if 'LCSC' not in df.columns:
        df['LCSC'] = ""
    if 'LCSC_Note' not in df.columns:
        df['LCSC_Note'] = ""

    total = len(df)
    print(f"[*] Processing {total} rows...\n")

    for idx, row in df.iterrows():
        ref = str(row[ref_col])
        val = str(row[val_col])
        footprint = str(row[foot_col]) if foot_col else ""
        mpn = str(row[mpn_col]) if mpn_col else ""

        print(f"[{idx+1}/{total}] {ref} | {val} | {footprint} ... ", end="", flush=True)

        res_str = find_lcsc_part(client, ref, val, footprint, mpn)
        
        try:
            # Clean the response in case it is wrapped in a Markdown code block
            clean_json = res_str.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            
            lcsc_code = data.get("lcsc_code", "")
            note = f"[{data.get('confidence', 'N/A')}] {data.get('note', '')}"
            
            df.at[idx, 'LCSC'] = lcsc_code
            df.at[idx, 'LCSC_Note'] = note
            print(f"-> {lcsc_code} ({data.get('confidence')})")
        except Exception as e:
            df.at[idx, 'LCSC_Note'] = f"Parse Error: {res_str[:50]}"
            print("-> [Parsing error]")

    # Save to a new file
    output_path = os.path.splitext(file_path)[0] + "_lcsc.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[+] Done! Enriched file saved to: {output_path}")

if __name__ == "__main__":
    main()