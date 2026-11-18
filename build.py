"""Build site/index.html from src/ (the single-file tool the team opens in the browser).
Run:  python3 build.py
"""
import pathlib
root = pathlib.Path(__file__).parent
src = (root / "src" / "shirt-reel-maker.src.html").read_text()
hands = (root / "src" / "hands.js").read_text()
assert "<!--HANDDATA-->" in src
out = src.replace("<!--HANDDATA-->", "<script>" + hands + "</script>", 1)
(root / "site").mkdir(exist_ok=True)
(root / "site" / "index.html").write_text(out)
print("built site/index.html", len(out) // 1024, "KB")
