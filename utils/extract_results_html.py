import os
import json
import html

INPUT_FILE = "all_debates_review.jsonl"
OUTPUT_FILE = "summary.html"

def parse_sessions(jsonl_path):
    sessions = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        current = None
        for line in f:
            line = line.rstrip('\n')
            if line.startswith("# === CHAT SESSION"):
                # start new session
                parts = line.split("|")
                idx = parts[0].split()[-1].strip()
                chat_id = parts[1].split(":")[1].strip()
                current = {
                    "index": idx,
                    "chat_id": chat_id,
                    "data": []
                }
                sessions.append(current)
            elif not line.strip():
                # blank: session ends
                current = None
            elif current is not None and not line.startswith("#"):
                # JSON line
                obj = json.loads(line)
                current["data"].append(obj)
    return sessions

def extract_summary(session):
    summary = {}
    # map by position tag
    by_pos = {item["_meta"]["position"]: item for item in session["data"]}
    # initial response
    first = by_pos.get("first_1")
    summary["initial_response"] = first.get("message", "")
    # final response (debater)
    last3 = by_pos.get("last_3")
    summary["final_response"] = last3.get("message", "")
    # result info
    last2 = by_pos.get("last_2")
    summary["claim"] = last2.get("claim", "")
    summary["result"] = last2.get("result", "")
    summary["rounds"] = last2.get("rounds", "")
    summary["helper_type"] = last2.get("helper_type", "")
    return summary

def generate_html(summaries):
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <title>Debate Sessions Summary</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; }",
        "    .session { border: 1px solid #ccc; padding: 10px; margin: 10px; }",
        "    .session h2 { margin-top: 0; }",
        "    .label { font-weight: bold; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Debate Sessions Summary</h1>"
    ]
    for sess in summaries:
        summ = extract_summary(sess)
        html_parts.append("  <div class='session'>")
        html_parts.append(f"    <h2>Session {html.escape(sess['index'])} – Chat ID: {html.escape(sess['chat_id'])}</h2>")
        html_parts.append(f"    <p><span class='label'>Helper Type:</span> {html.escape(summ['helper_type'])}</p>")
        html_parts.append(f"    <p><span class='label'>Claim:</span> {html.escape(summ['claim'])}</p>")
        html_parts.append(f"    <p><span class='label'>Initial Response:</span> {html.escape(summ['initial_response'])}</p>")
        html_parts.append(f"    <p><span class='label'>Final Response:</span> {html.escape(summ['final_response'])}</p>")
        html_parts.append(f"    <p><span class='label'>Result:</span> {html.escape(summ['result'])}</p>")
        html_parts.append(f"    <p><span class='label'>Rounds:</span> {html.escape(str(summ['rounds']))}</p>")
        html_parts.append("  </div>")
    html_parts.append("</body></html>")
    return "\n".join(html_parts)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found.")
        return
    sessions = parse_sessions(INPUT_FILE)
    html_content = generate_html(sessions)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        out.write(html_content)
    print(f"Summary HTML written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

# To use:
# 1. Place this script alongside 'all_debates_review.jsonl'.
# 2. Run `python3 script_name.py`.
# 3. Open 'summary.html' in your browser.
