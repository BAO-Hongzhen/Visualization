import os
import datetime
import dotenv

from scraping_utils import get_url, parse

dotenv.load_dotenv()

year = int(os.getenv('YEAR', 2024))
filename = os.getenv('FILENAME', "crawled-page-{year}.html").format(year=year)

if os.path.exists(filename):
    with open(filename, 'r', encoding='UTF8') as f:
        html_content = f.read()

else:
    url = os.getenv('URL').replace('${YEAR}', str(year))
    html_content = get_url(url, filename)

tree = parse(html_content, 'html')

rows = tree.xpath(os.getenv('ROW_XPATH'))

# Process tide data similar to tides_csv.py
data = []
row_num = 0

for row in tree.xpath(os.getenv('ROW_XPATH')):
    columns = row.xpath(os.getenv('COL_XPATH'))
    columns = [column.text_content().strip() for column in columns]
    row_string = " ".join(columns).strip()
    
    # Skip empty rows
    if row_string.strip() == "":
        continue
    
    row_num += 1
    
    # Skip header or invalid rows
    if len(columns) < 3 or not columns[0].isdigit():
        continue
    
    try:
        month = int(columns[0])
        day = int(columns[1])
        
        # Process time and value pairs
        for i in range(2, len(columns), 2):
            if i+1 < len(columns) and columns[i] != "" and len(columns[i]) >= 4:
                # Extract time in HHMM format
                time_str = columns[i]
                if len(time_str) >= 4 and time_str.isdigit():
                    hour = int(time_str[:2])
                    minute = int(time_str[2:])
                    
                    dt = datetime.datetime(year, month, day, hour, minute)
                    value = columns[i+1]
                    
                    if value:  # Make sure value is not empty
                        data.append((dt, value))

    except (ValueError, IndexError) as e:
        # Skip problematic rows
        continue


# Create CSV file，始终保存在脚本同目录
csv_filename = os.path.join(os.path.dirname(__file__), 'tides_processed.csv')

with open(csv_filename, 'w') as f:
    f.write('datetime,tide_level\n')  # Header
    for record in data:
        f.write(f'{record[0].strftime("%Y-%m-%d %H:%M")},{record[1]}\n')
