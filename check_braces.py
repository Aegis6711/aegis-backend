with open("rendered_page.html", "r", encoding="utf-8") as f:
    content = f.read()

scripts = []
idx = 0
while True:
    start = content.find("<script>", idx)
    if start == -1:
        break
    end = content.find("</script>", start)
    scripts.append(content[start + len("<script>"):end])
    idx = end + len("</script>")

big_script = scripts[-1]
lines = big_script.split("\n")

brace = 0
paren = 0
for i, line in enumerate(lines, start=1):
    brace += line.count("{") - line.count("}")
    paren += line.count("(") - line.count(")")
    print(f"{i}: brace={brace} paren={paren} | {line.strip()[:80]}")