import os
import subprocess
import markdown

def convert_to_pdf():
    md_path = os.path.join(os.path.dirname(__file__), "Glance_Submission_Report.md")
    html_path = os.path.join(os.path.dirname(__file__), "Glance_Submission_Report_temp.html")
    pdf_path = os.path.join(os.path.dirname(__file__), "Glance_Submission_Report.pdf")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_content = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"]
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Glance ML Internship Assignment - Submission Report</title>
    <style>
        @page {{
            size: A4;
            margin: 20mm 20mm 20mm 20mm;
        }}
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #222222;
            background-color: #ffffff;
            margin: 0;
            padding: 0;
        }}
        h1 {{
            font-size: 20pt;
            color: #1a1a2e;
            margin-top: 0;
            margin-bottom: 5px;
            border-bottom: 2px solid #e94560;
            padding-bottom: 8px;
        }}
        h2 {{
            font-size: 14pt;
            color: #0f3460;
            margin-top: 24px;
            margin-bottom: 12px;
            border-bottom: 1px solid #dddddd;
            padding-bottom: 4px;
        }}
        h3 {{
            font-size: 12pt;
            color: #e94560;
            margin-top: 18px;
            margin-bottom: 8px;
        }}
        p, li {{
            margin-bottom: 8px;
        }}
        code {{
            font-family: 'Consolas', 'Courier New', monospace;
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 4px;
            font-size: 9.5pt;
            color: #c7254e;
        }}
        pre {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 12px;
            overflow-x: auto;
            page-break-inside: avoid;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
            color: #333333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 9.5pt;
            page-break-inside: avoid;
        }}
        th, td {{
            padding: 8px 10px;
            border: 1px solid #cccccc;
            text-align: left;
            vertical-align: top;
        }}
        th {{
            background-color: #0f3460;
            color: #ffffff;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        blockquote {{
            border-left: 4px solid #e94560;
            margin: 14px 0;
            padding: 8px 16px;
            background-color: #fff8f8;
            color: #555555;
        }}
        hr {{
            border: 0;
            border-top: 1px solid #eeeeee;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    # Try Edge or Chrome headless PDF print
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    edge_path_alt = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

    browser_exe = None
    if os.path.exists(edge_path):
        browser_exe = edge_path
    elif os.path.exists(chrome_path):
        browser_exe = chrome_path
    elif os.path.exists(edge_path_alt):
        browser_exe = edge_path_alt

    if not browser_exe:
        print("ERROR: Neither Edge nor Chrome found for PDF export.")
        return

    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]
    print(f"Running command with {browser_exe}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(pdf_path):
        print(f"SUCCESS: PDF generated at {pdf_path}")
        try:
            os.remove(html_path)
        except Exception:
            pass
    else:
        print(f"PDF generation failed or returned error: {res.stderr} {res.stdout}")

if __name__ == "__main__":
    convert_to_pdf()
