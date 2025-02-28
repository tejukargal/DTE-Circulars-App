import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
from datetime import datetime
import time
import warnings
import urllib3
from datetime import datetime
import base64

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# URLs dictionary
urls = {
    "Departmental Circulars": "https://dtek.karnataka.gov.in/info-4/Departmental+Circulars/kn",
    "DVP Circulars": "https://dtek.karnataka.gov.in/page/Circulars/DVP/kn",
    "ACM-Polytechnic Circulars": "https://dtek.karnataka.gov.in/page/Circulars/ACM-Polytechnic/kn",
    "Exam Circulars": "https://dtek.karnataka.gov.in/page/Circulars/Exam/kn"
}

def display_pdf(url):
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
        
        base64_pdf = base64.b64encode(response.content).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error("PDF ಲೋಡ್ ಮಾಡಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ")

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%d/%m/%Y')
    except:
        return datetime.strptime('01/01/2000', '%d/%m/%Y')

def scrape_table(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=30)
        response.raise_for_status()
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table')
            
            if table:
                data = []
                rows = table.find_all('tr')[1:]  # Skip header row
                
                progress_bar = st.progress(0)
                total_rows = len(rows)
                
                for idx, row in enumerate(rows):
                    cols = row.find_all('td')
                    
                    # Skip rows with insufficient columns
                    if len(cols) < 4:
                        continue
                        
                    # Get basic data (first 3 columns)
                    date = cols[0].text.strip() if cols[0].text.strip() else None
                    ref_number = cols[1].text.strip() if cols[1].text.strip() else None
                    subject = cols[2].text.strip() if cols[2].text.strip() else None
                    
                    # Skip row if any essential column is missing
                    if not all([date, ref_number, subject]):
                        continue
                    
                    # Look for hyperlink in 4th or 5th column
                    download_url = ''
                    link_text = ''
                    
                    # Function to extract link from column
                    def get_link_from_column(col):
                        link_element = col.find('a')
                        if link_element:
                            href = link_element.get('href', '')
                            if href and not href.startswith('http'):
                                href = 'https://dtek.karnataka.gov.in' + href
                            return href, link_element.text.strip()
                        return None, col.text.strip()
                    
                    # Try 4th column first
                    if len(cols) >= 4:
                        download_url, link_text = get_link_from_column(cols[3])
                    
                    # If no link found and 5th column exists, try 5th column
                    if not download_url and len(cols) >= 5:
                        download_url, link_text = get_link_from_column(cols[4])
                    
                    # Skip row if no link text found
                    if not link_text:
                        continue
                    
                    data.append({
                        'date_obj': parse_date(date),
                        'ದಿನಾಂಕ': date,
                        'ಆದೇಶ ಸಂಖ್ಯೆ': ref_number,
                        'ವಿಷಯ': subject,
                        'ಸುತ್ತೋಲೆ': link_text,
                        'url': download_url
                    })
                    
                    progress_bar.progress((idx + 1) / total_rows)
                    time.sleep(0.01)
                
                if not data:
                    st.error("ಯಾವುದೇ ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ")
                    return None
                
                df = pd.DataFrame(data)
                df = df.sort_values('date_obj', ascending=False)
                df = df.head(50)
                df.insert(0, 'ಕ್ರಮ ಸಂಖ್ಯೆ', range(1, len(df) + 1))
                return df
            else:
                st.error("ಕೋಷ್ಟಕ ಕಂಡುಬಂದಿಲ್ಲ")
                return None
                
    except Exception as e:
        st.error(f"ದೋಷ ಸಂಭವಿಸಿದೆ: {str(e)}")
        return None

def create_excel_file(df):
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Create a copy of dataframe without technical columns
        excel_df = df.drop(['date_obj', 'url'], axis=1)
        excel_df.to_excel(writer, sheet_name='Sheet1', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'font_size': 10,
            'text_wrap': True,
            'align': 'center',
            'valign': 'vcenter',
            'bg_color': '#D9EAD3'
        })
        
        cell_format = workbook.add_format({
            'font_size': 9,
            'text_wrap': True,
            'valign': 'vcenter'
        })
        
        link_format = workbook.add_format({
            'font_size': 9,
            'font_color': 'blue',
            'underline': True
        })
        
        watermark_format = workbook.add_format({
            'font_size': 10,
            'italic': True,
            'font_color': '#888888',
            'align': 'right'
        })
        
        # Set column widths
        worksheet.set_column('A:A', 6)   # Sl No
        worksheet.set_column('B:B', 12)  # Date
        worksheet.set_column('C:C', 20)  # Reference No
        worksheet.set_column('D:D', 50)  # Subject
        worksheet.set_column('E:E', 15)  # Circular link
        
        # Format headers
        for col_num, value in enumerate(excel_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Write data with formatting and hyperlinks
        for row_num in range(len(df)):
            # Write regular columns
            worksheet.write(row_num + 1, 0, excel_df.iloc[row_num]['ಕ್ರಮ ಸಂಖ್ಯೆ'], cell_format)
            worksheet.write(row_num + 1, 1, excel_df.iloc[row_num]['ದಿನಾಂಕ'], cell_format)
            worksheet.write(row_num + 1, 2, excel_df.iloc[row_num]['ಆದೇಶ ಸಂಖ್ಯೆ'], cell_format)
            worksheet.write(row_num + 1, 3, excel_df.iloc[row_num]['ವಿಷಯ'], cell_format)
            
            # Write hyperlink in the last column
            if df.iloc[row_num]['url']:
                worksheet.write_url(
                    row_num + 1, 4,
                    df.iloc[row_num]['url'],
                    link_format,
                    string=df.iloc[row_num]['ಸುತ್ತೋಲೆ']
                )
            else:
                worksheet.write(row_num + 1, 4, df.iloc[row_num]['ಸುತ್ತೋಲೆ'], cell_format)
        
        # Add watermark at the bottom of the sheet
        last_row = len(df) + 3
        worksheet.merge_range(last_row, 0, last_row, 4, "by Teju SMP", watermark_format)
    
    return output.getvalue()

def add_watermark():
    # Add watermark using CSS
    st.markdown(
        """
        <style>
        .watermark {
            position: fixed;
            bottom: 10px;
            right: 10px;
            opacity: 0.7;
            z-index: 999;
            color: #888;
            font-style: italic;
        }
        </style>
        <div class="watermark">by Teju SMP</div>
        """,
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title="ಕರ್ನಾಟಕ ತಾಂತ್ರಿಕ ಶಿಕ್ಷಣ ಇಲಾಖೆ - ಸುತ್ತೋಲೆಗಳು", layout="wide")
    
    # Add watermark to UI
    add_watermark()
    
    st.title("ಕರ್ನಾಟಕ ತಾಂತ್ರಿಕ ಶಿಕ್ಷಣ ಇಲಾಖೆ - ಸುತ್ತೋಲೆಗಳು")
    
    # Sub-header with watermark
    st.markdown("<h3 style='text-align: center;'>ಸುತ್ತೋಲೆ ಡೌನ್‌ಲೋಡರ್ <span style='font-size: 12px; font-style: italic; color: #888;'>by Teju SMP</span></h3>", 
                unsafe_allow_html=True)
    
    selected_url_name = st.selectbox(
        "ಸುತ್ತೋಲೆ ವಿಧವನ್ನು ಆಯ್ಕೆ ಮಾಡಿ:",
        list(urls.keys())
    )
    
    if st.button("ಡೇಟಾ ಪಡೆಯಿರಿ"):
        with st.spinner("ಡೇಟಾ ಪಡೆಯಲಾಗುತ್ತಿದೆ..."):
            df = scrape_table(urls[selected_url_name])
            
            if df is not None and not df.empty:
                # Store the full dataframe in session state
                st.session_state['full_df'] = df
                
                # Create preview dataframe - remove technical columns
                preview_df = df.drop(['date_obj', 'url'], axis=1)
                
                # Show preview with clickable links
                st.subheader("ಡೇಟಾ ಪ್ರಿವ್ಯೂ")
                st.dataframe(
                    preview_df,
                    height=400,
                    column_config={
                        "ಸುತ್ತೋಲೆ": st.column_config.LinkColumn("ಸುತ್ತೋಲೆ")
                    },
                    hide_index=True
                )
                
                # Handle PDF preview
                if 'selected_row' in st.session_state:
                    selected_url = st.session_state['full_df'].iloc[st.session_state['selected_row']]['url']
                    if selected_url:
                        st.subheader("PDF ಪ್ರಿವ್ಯೂ")
                        display_pdf(selected_url)
                
                # Create Excel file
                excel_data = create_excel_file(df)
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"ಸುತ್ತೋಲೆಗಳು_{current_time}.xlsx"
                
                st.download_button(
                    label="Excel ಫೈಲ್ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
                    data=excel_data,
                    file_name=file_name,
                    mime="application/vnd.ms-excel"
                )
                
                st.info(f"ಒಟ್ಟು {len(df)} ದಾಖಲೆಗಳು ಪಡೆಯಲಾಗಿದೆ")
    
    # Footer watermark
    st.markdown(
        "<div style='text-align: center; color: #888; padding: 20px; font-style: italic;'>©2025 Teju SMP</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()