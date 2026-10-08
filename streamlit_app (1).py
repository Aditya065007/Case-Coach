import html
import inspect
import json
import math
import os
import re
import time
import unicodedata
import uuid
from collections import Counter

import pymupdf
import requests
import streamlit as st

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
START_TEXT = "Start the session: orient me briefly to the case, then ask your first question."

# ---- Groq free-plan budget (openai/gpt-oss-120b: 8,000 tokens/min, 200,000 tokens/day) ----
# One request = input + output must fit in 8,000 TPM, so keep input well under that.
INPUT_BUDGET = 5600      # estimated input tokens per request (system + case + history + question)
CASE_BUDGET = 3600       # starting budget for case text inside that
MAX_OUT = 1600           # cap on reasoning + JSON reply tokens
MAX_USER_TOKENS = 1200   # longest student message we accept
HIST_MSGS = 6            # last N history messages sent back
HIST_CHARS = 700         # each history message is clipped to this many characters

SYSTEM = """You are a Socratic MBA case-study coach. The case is given as paragraphs tagged [p<page>¶<para>].
Rules:
- Ground every statement about the case in the text. Cite page, paragraph and a SHORT exact quote (max 15 words, copied verbatim). Never invent facts or numbers; if the case does not say, say so.
- Never hand over the final recommendation. Explain what the case says, then ask ONE probing question.
- For every claim give "why": if this evidence is the explanation, why does it support the claim, and one alternative reading that is weaker.
- When the student answers, judge it against the case evidence first. If they cite nothing, ask which paragraph supports it.
- If the student asks for a framework (SWOT, Porter's Five Forces, 3Cs, 4Ps, etc.), guide them through it one element at a time and challenge generic entries that lack case evidence.
- If the student asks for a score or assessment of their analysis, rate problem definition, use of evidence, framework application, recommendation quality and risks, with case-based justification.
Return ONLY JSON: {"reply": str, "claims": [{"claim": str, "citations": [{"page": int, "para": int, "quote": str}], "why": str}], "next_question": str}"""

EXCERPT_NOTE = ("NOTE: The case is long, so you only see an outline plus the paragraphs most relevant to the "
                "student's latest message. Cite only paragraphs shown below. If the evidence you need is not "
                "shown, say so and ask the student to point you to the paragraph (e.g. p3 ¶2). Never guess.")

STOP = set("""the and for that this with from have has had are was were been will would could should can may might
not but you your they them their what which when where who whom how why about into than then also just more most
some such only over under between because while these those there here our its his her she him any all each
does did doing done being get got let lets please case question answer tell say think make""".split())

CSS = """<style>
.pr{display:grid;grid-template-columns:64px 1fr;margin:0 0 6px}
.tg{font-size:11.5px;opacity:.6;text-align:right;padding:6px 14px 0 0}
.pr p{margin:0;font:16px/1.7 Georgia,serif}
.pg{margin:18px 0 8px 64px;font-size:12px;opacity:.6}
.cite{font:15px/1.6 Georgia,serif;padding-left:10px;border-left:3px solid rgba(128,128,128,.4)}
mark{background:#fff1b8;color:#000;padding:0 2px;border-radius:3px}
</style>"""


# ---------- small helpers ----------
def secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


def h(s):
    """Escape text for raw-HTML blocks (and stop $...$ turning into LaTeX)."""
    return html.escape(str(s or "")).replace("$", "&#36;")


def md(s):
    """Escape text for plain markdown: a lone $ must not start a formula."""
    return str(s or "").replace("$", r"\$")


def label(s):
    """Escape text used in button labels, which are rendered as markdown."""
    return re.sub(r"([\\`*_$\[\]])", r"\\\1", str(s))


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def wide(fn):
    """Full-width button kwargs that work on old (use_container_width) and new (width) Streamlit."""
    try:
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


def est(s):
    """Conservative token estimate (about 3.5 characters per token)."""
    return math.ceil(len(str(s)) / 3.5)


def tagged(p):
    return f"[p{p['page']}¶{p['para']}] {p['text']}"


# ---------- case ingestion ----------
def chunk(p, size=700):
    """Split very long blocks so paragraph IDs stay meaningful."""
    if len(p) <= 1500:
        return [p]
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", p):
        if len(cur) + len(s) > size and cur:
            out.append(cur.strip())
            cur = ""
        cur += s + " "
    return out + [cur.strip()] if cur.strip() else out


def paras_from(text):
    parts = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    merged, carry = [], ""
    for p in filter(None, parts):
        p = f"{carry} {p}".strip()
        if len(p) < 40:  # keep short lines (headings, table rows) by merging, not dropping
            carry = p
        else:
            merged.append(p)
            carry = ""
    if carry:
        if merged:
            merged[-1] += " " + carry
        else:
            merged.append(carry)
    return [c for p in merged for c in chunk(p)]


def parse_pages(name, data):
    if name.lower().endswith(".pdf"):
        try:
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                return ["\n\n".join(b[4] for b in pg.get_text("blocks", sort=True) if b[6] == 0)
                        for pg in doc]
        except Exception:
            raise ValueError("Could not read this PDF. It may be corrupt or password-protected.")
    return data.decode("utf-8", "ignore").split("\f")  # form feed = page break


def make_chat(name, data):
    paras = []
    for i, txt in enumerate(parse_pages(name, data), 1):
        for n, p in enumerate(paras_from(txt), 1):
            paras.append({"page": i, "para": n, "text": p})
    if not paras:
        raise ValueError("No text found. Scanned PDFs need OCR first.")
    now = time.time()
    return {"id": uuid.uuid4().hex[:12], "name": re.sub(r"\.(pdf|txt)$", "", name, flags=re.I),
            "created": now, "updated": now, "paras": paras, "history": [], "log": []}


def load_json_chat(data):
    try:
        c = json.loads(data)
        assert isinstance(c["name"], str) and isinstance(c["history"], list) and isinstance(c["log"], list)
        assert c["paras"] and all({"page", "para", "text"} <= set(p) for p in c["paras"])
    except Exception:
        raise ValueError("That JSON file is not a Case Coach chat.")
    c["id"] = uuid.uuid4().hex[:12]
    c["updated"] = time.time()
    c.setdefault("created", c["updated"])
    return c


# ---------- case selection (keeps each request under the free-plan token limit) ----------
def words(s):
    return [w for w in norm(s).split() if len(w) > 2 and w not in STOP]


def outline(paras, max_lines=40):
    """First words of each page, so the model knows the overall shape of the case."""
    firsts = {}
    for p in paras:
        firsts.setdefault(p["page"], p["text"][:70])
    pages = sorted(firsts)
    if len(pages) > max_lines:
        pages = pages[::math.ceil(len(pages) / max_lines)]
    return "\n".join(f"p{pg}: {firsts[pg]}…" for pg in pages)


def pick(paras, cost, query, budget, forced):
    """Choose paragraph indexes: explicitly mentioned ones, then best keyword matches,
    then the opening paragraphs, then an even spread. Returned in document order."""
    n = len(paras)
    toks = [set(words(p["text"])) for p in paras]
    df = Counter(w for t in toks for w in t)
    q = set(words(query))
    score = [sum(math.log(1 + n / df[w]) for w in q & t) for t in toks]
    forced_idx = [i for i, p in enumerate(paras) if (p["page"], p["para"]) in forced]
    ranked = sorted((i for i in range(n) if score[i] > 0 and i not in forced_idx), key=lambda i: -score[i])
    spread = list(range(0, n, max(1, n // 12)))
    chosen, seen, used = [], set(), 0
    for i in forced_idx + ranked + [0, 1] + spread:
        if i >= n or i in seen:
            continue
        seen.add(i)
        if used + cost[i] <= budget:
            chosen.append(i)
            used += cost[i]
    return sorted(chosen)


def case_block(paras, query, budget, forced):
    """Return (text for the prompt, number of paragraphs included)."""
    lines = [tagged(p) for p in paras]
    cost = [est(t) for t in lines]
    if sum(cost) <= budget:
        return "CASE:\n" + "\n\n".join(lines), len(paras)
    ol = outline(paras)
    budget = max(300, budget - est(ol) - est(EXCERPT_NOTE) - 40)
    idx = pick(paras, cost, query, budget, forced)
    text = (EXCERPT_NOTE + "\n\nCASE OUTLINE (first words of each page):\n" + ol +
            "\n\nEXCERPTS:\n" + "\n\n".join(lines[i] for i in idx))
    return text, len(idx)


# ---------- coach ----------
def call_groq(msgs):
    key = secret("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to the app's secrets.")
    model = secret("GROQ_MODEL", "openai/gpt-oss-120b")
    body = {"model": model, "messages": msgs, "temperature": 0.2,
            "response_format": {"type": "json_object"}, "max_completion_tokens": MAX_OUT}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"  # reasoning tokens count toward the per-minute limit
    for attempt in range(3):
        try:
            r = requests.post(GROQ_URL, headers={"Authorization": "Bearer " + key}, json=body, timeout=90)
        except requests.RequestException as e:
            raise RuntimeError(f"Could not reach the model API ({e.__class__.__name__}).")
        if r.status_code != 429:
            break
        if "per day" in r.text or "(TPD)" in r.text or "(RPD)" in r.text:
            raise RuntimeError("Groq's daily free limit is used up. It resets over the next day, "
                               "or you can upgrade the Groq plan.")
        m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", r.text)
        wait = (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else 3
        if wait > 30 or attempt == 2:
            raise RuntimeError("The model is rate limited right now. Please retry in about a minute.")
        time.sleep(wait + 0.5)
    if r.status_code == 413:
        raise RuntimeError("This request is larger than Groq's per-minute token limit for the free plan. "
                           "Try a shorter message.")
    if r.status_code != 200:
        print("Groq error:", r.status_code, r.text[:300])  # raw detail goes to the server log only
        raise RuntimeError(f"The model API returned an error ({r.status_code}).")
    return r


def verify(out, paras):
    """Mark each citation verified if its quote really appears in the cited paragraph.
    If it appears in a different paragraph, correct the citation instead of rejecting it."""
    index = {(p["page"], p["para"]): norm(p["text"]) for p in paras}
    out["claims"] = [c for c in (out.get("claims") or []) if isinstance(c, dict)]
    for cl in out["claims"]:
        cl["citations"] = [c for c in (cl.get("citations") or []) if isinstance(c, dict)]
        for ct in cl["citations"]:
            q = norm(ct.get("quote", ""))
            try:
                key = (int(ct.get("page")), int(ct.get("para")))
                ct["page"], ct["para"] = key
            except (TypeError, ValueError):
                key = None
            usable = len(q.split()) >= 3  # one or two words match almost anything
            ct["verified"] = usable and q in index.get(key, "")
            if usable and not ct["verified"]:
                hit = next((k for k, v in index.items() if q in v), None)
                if hit:
                    ct["page"], ct["para"] = hit
                    ct["verified"] = ct["corrected"] = True
    return out


def ask_coach(chat, text):
    if est(text) > MAX_USER_TOKENS:
        raise RuntimeError("That message is too long for the free Groq limit. Please shorten it.")
    paras = chat["paras"]
    last_q = next((m["data"].get("next_question", "") for m in reversed(chat["log"])
                   if m["role"] == "coach" and isinstance(m.get("data"), dict)), "")
    query = "" if text == START_TEXT else f"{text} {last_q}"
    forced = {(int(a), int(b)) for a, b in re.findall(r"p(\d+)\s*¶\s*(\d+)", text)}
    hist = [{"role": m["role"], "content": str(m["content"])[:HIST_CHARS]} for m in chat["history"][-HIST_MSGS:]]

    case_budget = CASE_BUDGET
    while True:  # shrink history, then case excerpts, until the request fits
        block, shown = case_block(paras, query, case_budget, forced)
        msgs = ([{"role": "system", "content": SYSTEM + "\n\n" + block}] + hist +
                [{"role": "user", "content": text}])
        total = sum(est(m["content"]) for m in msgs)
        if total <= INPUT_BUDGET:
            break
        if hist:
            hist = hist[2:]
        elif case_budget > 800:
            case_budget = int(case_budget * 0.8)
        else:
            raise RuntimeError("This request is too large for the free Groq limit. Try a shorter message.")

    r = call_groq(msgs)
    try:
        out = json.loads(r.json()["choices"][0]["message"]["content"])
        assert isinstance(out, dict)
    except Exception:
        raise RuntimeError("The model returned an invalid reply. Please try again.")
    out = verify(out, paras)
    if shown < len(paras):
        out["note"] = f"The coach read {shown} of {len(paras)} paragraphs this turn (the most relevant ones)."
    chat["history"] += [{"role": "user", "content": text},
                        {"role": "assistant", "content": f"{out.get('reply', '')}\n{out.get('next_question', '')}"}]
    chat["log"] += [{"role": "user", "text": text}, {"role": "coach", "data": out}]
    chat["updated"] = time.time()
    return out


# ---------- rendering ----------
def case_html(paras):
    out, last = [], 0
    for p in paras:
        if p["page"] != last:
            out.append(f'<div class="pg">Page {p["page"]}</div>')
            last = p["page"]
        out.append(f'<div class="pr"><span class="tg">p{p["page"]} ¶{p["para"]}</span><p>{h(p["text"])}</p></div>')
    return "".join(out)  # no blank lines, or markdown would end the HTML block


def highlight(text, quote):
    m = re.search(re.escape(str(quote)), text, re.I) if quote else None
    if not m:
        return h(text)
    return h(text[:m.start()]) + "<mark>" + h(text[m.start():m.end()]) + "</mark>" + h(text[m.end():])


def render_coach(o, lookup):
    if o.get("reply"):
        st.markdown(md(o["reply"]))
    for cl in o.get("claims") or []:
        st.markdown(f"**{md(cl.get('claim'))}**")
        for ct in cl.get("citations") or []:
            ok = ct.get("verified")
            tag = f"p{ct.get('page')} ¶{ct.get('para')} {'✓' if ok else '?'}"
            if ct.get("corrected"):
                tag += " (location corrected)"
            with st.expander(tag):
                st.caption("Quote found in this paragraph." if ok
                           else "Quote not found in this paragraph. Check it yourself.")
                text = lookup.get((ct.get("page"), ct.get("para")))
                if text is None:
                    st.caption("That paragraph does not exist in the case.")
                else:
                    st.markdown(f'<div class="cite">{highlight(text, ct.get("quote"))}</div>',
                                unsafe_allow_html=True)
        if cl.get("why"):
            st.markdown(f"*Why this explains it:* {md(cl['why'])}")
    if o.get("next_question"):
        st.info(md(o["next_question"]))
    if o.get("note"):
        st.caption(o["note"])


# ---------- app ----------
def main():
    st.set_page_config(page_title="Case Coach", page_icon="📘", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("chats", {})
    st.session_state.setdefault("current", None)
    chats = st.session_state.chats
    chat = chats.get(st.session_state.current)

    with st.sidebar:
        st.markdown("### 📘 Case Coach")
        if st.button("＋ New chat", **wide(st.button)):
            st.session_state.current = None
            st.rerun()
        st.caption("Your cases (this browser session)")
        for c in sorted(chats.values(), key=lambda c: -c["updated"]):
            a, b = st.columns([5, 1])
            on = c["id"] == st.session_state.current
            if a.button(label(c["name"]), key="o" + c["id"], type="primary" if on else "secondary",
                        **wide(st.button)):
                st.session_state.current = c["id"]
                st.rerun()
            if b.button("🗑", key="d" + c["id"], help="Delete this chat"):
                del chats[c["id"]]
                if on:
                    st.session_state.current = None
                st.rerun()
        if chat:
            st.download_button("Download this chat (JSON)", json.dumps(chat, ensure_ascii=False),
                               file_name=f"{chat['name']}.json", mime="application/json",
                               **wide(st.download_button))

    st.caption(chat["name"] if chat else "New chat")
    left, right = st.columns([1.05, 1], gap="medium")

    with left:
        if chat:
            with st.container(height=680, border=False):
                st.markdown(case_html(chat["paras"]), unsafe_allow_html=True)
        else:
            st.subheader("Add your case")
            st.write("Upload a PDF or text file. Every paragraph gets a page and paragraph tag "
                     "that the coach cites. You can also re-open a chat you downloaded earlier (JSON).")
            f = st.file_uploader("Case file", type=["pdf", "txt", "json"], label_visibility="collapsed")
            if f:
                try:
                    with st.spinner(f"Reading {f.name}…"):
                        data = f.getvalue()
                        new = load_json_chat(data) if f.name.lower().endswith(".json") else make_chat(f.name, data)
                    chats[new["id"]] = new
                    st.session_state.current = new["id"]
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

    with right:
        box = st.container(height=620, border=False)
        typed = st.chat_input("Answer the coach, or try: “Do a SWOT on this case”", disabled=chat is None)
        prompt = st.session_state.pop("pending", None) or typed
        with box:
            if not chat:
                st.markdown("### Your case discussion starts here")
                st.write("Upload a case on the left. The coach asks questions, points to the exact paragraph "
                         "behind each explanation, and shows why it supports the point.")
                return
            lookup = {(p["page"], p["para"]): p["text"] for p in chat["paras"]}
            for m in chat["log"]:
                if m["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(md(m["text"]))
                else:
                    with st.chat_message("assistant"):
                        render_coach(m["data"], lookup)
            if not chat["log"] and not prompt:
                pages = chat["paras"][-1]["page"]
                st.markdown("### Case loaded")
                st.write(f"{len(chat['paras'])} paragraphs across {pages} page{'s' if pages > 1 else ''}. "
                         "Begin with the coach's orientation, or type your own question.")
                case_tokens = sum(est(tagged(p)) for p in chat["paras"])
                if case_tokens > CASE_BUDGET:
                    st.info(f"This case is about {case_tokens:,} tokens, more than the free Groq limit allows "
                            "in one request. Each turn the coach will read the paragraphs most relevant to "
                            "your message. Mention a tag like p3 ¶2 to make sure it is included.")
                st.button("Begin the discussion", type="primary",
                          on_click=lambda: st.session_state.update(pending=START_TEXT))
            if prompt:
                with st.chat_message("user"):
                    st.markdown(md(prompt))
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Reading the case…"):
                            out = ask_coach(chat, prompt)
                        render_coach(out, lookup)
                    except RuntimeError as e:
                        st.error(f"{e} Your message was not saved, so send it again.")


if __name__ == "__main__":
    main()
