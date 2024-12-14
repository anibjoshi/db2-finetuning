import os
from bs4 import BeautifulSoup
import pandas as pd
import json

# Folder containing HTML files
input_folder = "src/data/webpages"  # Replace with your folder path
output_folder = "src/data/raw"  # Replace with your output folder path

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each HTML file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".html"):
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.jsonl")
        
        with open(input_file, "r", encoding="utf-8") as file:
            html_content = file.read()
        
        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Find the table in the HTML (modify as needed for specific tables)
        table = soup.find("table", {"id": "sqlmsg__sqlmsg"})  # Adjust table ID or attributes if necessary
        
        if table:
            # Extract table data
            table_data = []
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                row_data = []
                for cell in cells:
                    text = " ".join(cell.stripped_strings)
                    row_data.append(text)
                if row_data:
                    table_data.append(row_data)
            
            # Dynamically determine the number of columns
            max_columns = max(len(row) for row in table_data)
            
            # Ensure we only process the first four columns
            processed_data = [row[:4] for row in table_data]
            
            # Define specific column names
            column_names = ["id", "message", "explanation", "response"]
            
            # Convert to a pandas DataFrame
            df = pd.DataFrame(processed_data, columns=column_names)
            
            # Save the DataFrame as JSON Lines with the version field
            with open(output_file, "w", encoding="utf-8") as jsonl_file:
                for record in df.to_dict(orient="records"):
                    record["version"] = "12.1"  # Add the version field to each record
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            print(f"Processed '{filename}' and saved as JSONL to '{output_file}'")
        else:
            print(f"No table found in '{filename}'")

print("All files processed.")
