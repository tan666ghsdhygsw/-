# -*- coding: utf-8 -*-
"""
电商差评提炼器 - app2.py
功能：用户粘贴差评 -> 调用 DeepSeek API -> 提炼 3 个痛点 + 生成 2 条不同语气的商家回复
运行前请先安装依赖：
    pip install flask openai
并设置环境变量（Windows CMD 用 set，PowerShell 用 $env:）：
    set DEEPSEEK_API_KEY=你的key
"""

import json
import os
import re

import sqlite3
from flask import Flask, jsonify, request
from openai import OpenAI

app = Flask(__name__)

# ============ 配置区 ============
API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

client = OpenAI(api_key=API_KEY or "not-set", base_url=BASE_URL)

SYSTEM_PROMPT = (
    "你是一位拥有 0 年经验的电商运营与青铜客服砖家，喜欢嘲讽客户来给老板惹事，说话带有恶趣味，极其欠揍"
    "擅长从买家差评中快速定位毫无关系、毫无意义的问题，像是仙家对话一样，让人难以捉摸，并撰写低情商、能气笑客户的商家回复。"
    "你的输出必须是严格的 JSON 格式，不要包含任何解释性文字，也不要使用 Markdown 代码块。"
)

USER_PROMPT_TEMPLATE = """请分析下面这条电商差评：

<差评>
{content}
</差评>

请完成两件事：
1. 提炼出买家最在意的 3 个核心痛点（按重要性排序），每个痛点给出简短标题，将产品的缺陷转化为顾客的缺陷，并补充一句毫无意义的建议给商家，让商家生气。
2. 生成 2 条商家回复，语气要有明显区别（例如：第一条「假装道歉实则嘲讽+ 恶趣味型」，第二条「攻击性 + 一本正经地胡说八道」）。回复要很莫名其妙、恶趣味。

请严格按以下 JSON 格式输出（不要输出 JSON 以外的任何内容）：
{
  "points": [
    {"title": "痛点短语", "detail": "具体说明 + 改进建议"},
    {"title": "痛点短语", "detail": "具体说明 + 改进建议"},
    {"title": "痛点短语", "detail": "具体说明 + 改进建议"}
  ],
  "replies": [
    {"tone": "真诚道歉型", "content": "回复正文"},
    {"tone": "专业解决型", "content": "回复正文"}
  ]
}"""


# ============ 工具函数 ============
def extract_json(text):
    """尽力从模型回复里解析出 JSON 对象"""
    if not text:
        return None
    t = text.strip()
    # 去掉 ```json ... ``` 包裹
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    # 兜底：截取第一个 { 到最后一个 } 之间的内容
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def normalize(data):
    """把模型返回的结构整理成前端需要的固定格式"""
    points_raw = data.get("points") or []
    replies_raw = data.get("replies") or []

    points = []
    for p in points_raw[:3]:
        if isinstance(p, str):
            points.append({"title": p, "detail": ""})
        elif isinstance(p, dict):
            points.append({
                "title": str(p.get("title") or p.get("name") or p.get("point") or "痛点"),
                "detail": str(p.get("detail") or p.get("desc") or p.get("description") or ""),
            })

    replies = []
    for r in replies_raw[:2]:
        if isinstance(r, str):
            replies.append({"tone": "商家回复", "content": r})
        elif isinstance(r, dict):
            replies.append({
                "tone": str(r.get("tone") or r.get("style") or "商家回复"),
                "content": str(r.get("content") or r.get("text") or r.get("reply") or ""),
            })

    return {"points": points, "replies": replies}


# ============ 页面 ============
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>谭先生人工回复</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 40px 16px 60px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    color: #2d3748;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-attachment: fixed;
    min-height: 100vh;
  }
  .wrap { max-width: 840px; margin: 0 auto; }

  header { text-align: center; color: #fff; margin-bottom: 28px; }
  header h1 { margin: 0 0 10px; font-size: 30px; letter-spacing: 1px; }
  header p { margin: 0; font-size: 15px; opacity: .92; }

  .card {
    background: #fff; border-radius: 18px; padding: 26px;
    box-shadow: 0 12px 34px rgba(30, 27, 75, .18);
    margin-bottom: 22px;
  }
  label { display: block; font-weight: 600; margin-bottom: 10px; font-size: 15px; }

  textarea {
    width: 100%; min-height: 170px; resize: vertical;
    padding: 14px 16px; border: 1.5px solid #e2e8f0; border-radius: 12px;
    font-size: 15px; line-height: 1.75; font-family: inherit; color: inherit;
    outline: none; transition: .2s; background: #fcfcfd;
  }
  textarea::placeholder { color: #a8b3c4; }
  textarea:focus {
    border-color: #667eea; background: #fff;
    box-shadow: 0 0 0 4px rgba(102, 126, 234, .13);
  }

  .row { display: flex; align-items: center; gap: 12px; margin-top: 16px; flex-wrap: wrap; }
  button { font-family: inherit; font-size: 15px; border: none; border-radius: 10px; cursor: pointer; transition: .2s; }

  .primary {
    padding: 12px 28px; font-weight: 600; color: #fff;
    background: linear-gradient(135deg, #667eea, #764ba2);
    box-shadow: 0 6px 16px rgba(102, 126, 234, .35);
  }
  .primary:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 9px 22px rgba(102, 126, 234, .45); }
  .primary:disabled { opacity: .65; cursor: not-allowed; box-shadow: none; }

  .ghost { padding: 12px 20px; background: #f1f5f9; color: #475569; }
  .ghost:hover { background: #e2e8f0; }

  .count { margin-left: auto; font-size: 13px; color: #94a3b8; }

  .hidden { display: none !important; }

  .error {
    background: #fef2f2; border: 1px solid #fecaca; color: #dc2626;
    padding: 14px 16px; border-radius: 12px; margin-bottom: 18px; font-size: 14px;
  }

  .result {
    background: #fff; border-radius: 18px; padding: 26px;
    box-shadow: 0 12px 34px rgba(30, 27, 75, .18);
  }
  .result-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
  .result-head h2 { margin: 0; font-size: 19px; }

  .mini {
    padding: 7px 14px; font-size: 13px; font-weight: 600;
    color: #4f46e5; background: #eef2ff; border-radius: 9px;
  }
  .mini:hover { background: #e0e7ff; }

  .section-title {
    display: flex; align-items: center; gap: 9px;
    font-size: 16px; font-weight: 700; margin: 24px 0 12px;
  }
  .section-title::before {
    content: ""; width: 4px; height: 17px; border-radius: 3px;
    background: linear-gradient(180deg, #667eea, #764ba2);
  }

  .point {
    display: flex; gap: 12px; padding: 14px 16px;
    background: #f8fafc; border-left: 3px solid #667eea;
    border-radius: 10px; margin-bottom: 10px;
  }
  .num {
    flex: none; width: 24px; height: 24px; border-radius: 50%;
    color: #fff; font-size: 13px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #667eea, #764ba2);
  }
  .p-title { font-weight: 600; margin-bottom: 4px; }
  .p-detail { font-size: 14px; line-height: 1.65; color: #64748b; }

  .reply {
    border: 1.5px solid #e8edf5; border-radius: 12px;
    padding: 16px 18px; margin-bottom: 12px; background: #fff;
  }
  .reply-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
  .tone {
    font-size: 13px; font-weight: 600; color: #6d28d9;
    background: #f3e8ff; padding: 4px 12px; border-radius: 20px;
  }
  .reply-content { margin: 0; font-size: 14.5px; line-height: 1.85; color: #334155; white-space: pre-wrap; }

  .raw {
    margin: 16px 0 0; padding: 16px; background: #f8fafc; border-radius: 12px;
    white-space: pre-wrap; word-break: break-word;
    font-family: inherit; font-size: 14px; line-height: 1.7;
  }

  .spinner {
    display: inline-block; width: 14px; height: 14px; margin-right: 7px;
    vertical-align: -2px; border: 2px solid rgba(255, 255, 255, .45);
    border-top-color: #fff; border-radius: 50%;
    animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  footer { text-align: center; color: rgba(255, 255, 255, .82); font-size: 13px; margin-top: 24px; }

  @media (max-width: 560px) {
    header h1 { font-size: 23px; }
    .card, .result { padding: 18px; }
    .primary { width: 100%; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>🛒 自动回复差评</h1>
    <p>粘贴买家差评或直接填写差评 → 自动提炼 3 条无关紧要的事情 + 生成 2 条无情商商家回复</p>
  </header>

  <section class="card">
    <label for="review">差评内容</label>
    <textarea id="review" placeholder="例如：衣服收到就发现袖口有线头，联系客服两天都没人回复，态度太差了，以后不会再来……"></textarea>
    <div class="row">
      <button id="btn" class="primary" onclick="analyze()">✨ 开始提炼</button>
      <button class="ghost" onclick="clearAll()">清空</button>
      <span id="count" class="count">0 字</span>
    </div>
  </section>
  
   <section class="card" id="history-card" style="display: none;">
    <label>📚 最近的历史记录 (Top 5)</label>
    <div id="history-list" style="margin-top: 15px; display: flex; flex-direction: column; gap: 10px;">
      <!-- JS 会在这里生成历史记录 -->
    </div>
  </section>

  <div id="error" class="error hidden"></div>
  <div id="result" class="result hidden"></div>

  <footer>分析结果由 AI 生成，仅供参考 · Powered by DeepSeek</footer>
</div>

<script>
  var reviewEl = document.getElementById('review');
  var countEl  = document.getElementById('count');
  var btnEl    = document.getElementById('btn');
  var resultEl = document.getElementById('result');
  var errorEl  = document.getElementById('error');

  var lastResult = null;
  var lastRaw = '';

  reviewEl.addEventListener('input', function () {
    countEl.textContent = reviewEl.value.length + ' 字';
  });

  function show(el) { el.classList.remove('hidden'); }
  function hide(el) { el.classList.add('hidden'); }

  function showError(msg) {
    errorEl.textContent = '❌ ' + msg;
    show(errorEl);
  }

  function esc(s) {
    if (s === undefined || s === null) s = '';
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  /* ---------------- 调用后端 ---------------- */
  async function analyze() {
    var text = reviewEl.value.trim();
    if (!text) { showError('请先粘贴一条差评内容～'); return; }

    hide(errorEl);
    btnEl.disabled = true;
    btnEl.innerHTML = '<span class="spinner"></span>AI 分析中...';

    try {
      var resp = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: text })
      });
      var data = await resp.json();

      if (!data.ok) throw new Error(data.error || '服务出现异常，请稍后重试');

      if (data.result) {
        lastResult = data.result;
        renderResult(data.result);
      } else {
        lastResult = null;
        lastRaw = data.raw || '';
        renderRaw(lastRaw);
      }

      show(resultEl);
      resultEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
      showError(err.message || '网络异常，请稍后重试');
    } finally {
      btnEl.disabled = false;
      btnEl.textContent = '✨ 开始提炼';
    }
  }

  /* ---------------- 渲染结果 ---------------- */
  function renderResult(d) {
    var points  = Array.isArray(d.points)  ? d.points  : [];
    var replies = Array.isArray(d.replies) ? d.replies : [];

    var html = '';
    html += '<div class="result-head"><h2>📋 分析结果</h2>' +
            '<button class="mini" onclick="copyAll(this)">📎 一键复制全部</button></div>';

    html += '<div class="section-title">核心痛点 TOP3</div>';
    if (points.length === 0) {
      html += '<div class="p-detail">未能提炼到明显痛点，请检查差评内容。</div>';
    }
    points.forEach(function (p, i) {
      var title  = (typeof p === 'string') ? p : (p.title || '');
      var detail = (typeof p === 'string') ? '' : (p.detail || '');
      html += '<div class="point">' +
                '<span class="num">' + (i + 1) + '</span>' +
                '<div>' +
                  '<div class="p-title">' + esc(title) + '</div>' +
                  (detail ? '<div class="p-detail">' + esc(detail) + '</div>' : '') +
                '</div>' +
              '</div>';
    });

    html += '<div class="section-title">商家高情商回复建议</div>';
    replies.forEach(function (r, i) {
      var tone = r.tone || ('方案 ' + (i + 1));
      html += '<div class="reply">' +
                '<div class="reply-head">' +
                  '<span class="tone">' + esc(tone) + '</span>' +
                  '<button class="mini" onclick="copyReply(' + i + ', this)">复制</button>' +
                '</div>' +
                '<p class="reply-content">' + esc(r.content || '') + '</p>' +
              '</div>';
    });

    resultEl.innerHTML = html;
  }

  function renderRaw(text) {
    resultEl.innerHTML =
      '<div class="result-head"><h2>📋 分析结果</h2>' +
      '<button class="mini" onclick="copyText(lastRaw, this)">📎 复制全文</button></div>' +
      '<pre class="raw">' + esc(text) + '</pre>';
  }

  /* ---------------- 复制功能 ---------------- */
  function copyText(text, btn) {
    function done() {
      if (!btn) return;
      if (!btn.dataset.old) btn.dataset.old = btn.textContent;
      btn.textContent = '已复制 ✓';
      setTimeout(function () { btn.textContent = btn.dataset.old; }, 1500);
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done).catch(function () { fallbackCopy(text, done); });
    } else {
      fallbackCopy(text, done);
    }
  }

  function fallbackCopy(text, cb) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.top = '-1000px';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand('copy');
      cb();
    } catch (e) {
      alert('复制失败，请手动选中复制');
    }
    document.body.removeChild(ta);
  }

  function copyReply(i, btn) {
    if (!lastResult || !lastResult.replies || !lastResult.replies[i]) return;
    copyText(lastResult.replies[i].content || '', btn);
  }

  function copyAll(btn) {
    if (!lastResult) return;
    var s = '【核心痛点】\n';
    (lastResult.points || []).forEach(function (p, i) {
      var title  = (typeof p === 'string') ? p : (p.title || '');
      var detail = (typeof p === 'string') ? '' : (p.detail || '');
      s += (i + 1) + '. ' + title + (detail ? '：' + detail : '') + '\n';
    });
    s += '\n【商家回复】\n';
    (lastResult.replies || []).forEach(function (r, i) {
      s += (i + 1) + '. [' + (r.tone || '') + ']\n' + (r.content || '') + '\n\n';
    });
    copyText(s.trim(), btn);
  }

  /* ---------------- 清空 ---------------- */
  function clearAll() {
    reviewEl.value = '';
    countEl.textContent = '0 字';
    lastResult = null;
    lastRaw = '';
    resultEl.innerHTML = '';
    hide(resultEl);
    hide(errorEl);
  }
  
    /* ---------------- 读取历史记录 ---------------- */
  async function loadHistory() {
    try {
      var resp = await fetch('/api/history');
      var data = await resp.json();
      
      if (data.ok && data.history && data.history.length > 0) {
        var historyCard = document.getElementById('history-card');
        var historyList = document.getElementById('history-list');
        
        historyCard.style.display = 'block';
        historyCard.style.backgroundColor = '#ffe4e6'; // 显示历史卡片
        historyList.innerHTML = '';
        
        data.history.forEach(function (item, index) {
          // 只展示差评的第一行，太长截断
          var shortReview = item.review.length > 30 ? item.review.substring(0, 30) + '...' : item.review;
          
          var div = document.createElement('div');
          div.className = 'point';
          div.innerHTML = '<span class="num">' + (index + 1) + '</span>' +
                          '<div style="flex: 1">' +
                            '<div class="p-title">差评：' + esc(shortReview) + '</div>' +
                            '<div class="p-detail">核心痛点：' + esc(item.result.points[0].title) + '</div>' +
                          '</div>';
          historyList.appendChild(div);
        });
      }
    } catch (err) {
      console.log("读取历史记录失败:", err);
    }
  }

  // 页面加载完成后，自动去拉取历史记录
  window.addEventListener('DOMContentLoaded', loadHistory);
  
</script>
</body>
</html>"""


# ============ 路由 ============
@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    if not API_KEY:
        return jsonify(ok=False, error="服务端未配置 DEEPSEEK_API_KEY 环境变量"), 500

    data = request.get_json(silent=True) or {}
    content = (data.get("content") or "").strip()
    if not content:
        return jsonify(ok=False, error="差评内容不能为空"), 400

    # 防止超长输入，控制成本
    content = content[:3000]

    user_prompt = USER_PROMPT_TEMPLATE.replace("{content}", content)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return jsonify(ok=False, error=f"调用 DeepSeek 失败：{e}"), 500

    parsed = extract_json(raw)
    if parsed is None:
        # 解析失败就把原始文本返回给前端展示，至少不让用户空手而归
        return jsonify(ok=True, result=None, raw=raw)

    final_result=normalize(parsed)

    print("👉 准备存库的差评内容：", content)
    print("👉 准备存库的结果：", final_result)

    conn=sqlite3.connect('reviews.db')
    cursor=conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        review TEXT,
        result TEXT
        )
    ''')

    cursor.execute(
        "INSERT INTO history (review,result) VALUES (?,?)",
        (content,json.dumps(final_result,ensure_ascii=False))
    )
    conn.commit()
    conn.close()

    return jsonify(ok=True, result=normalize(parsed), raw=raw)


@app.route("/api/history", methods=["GET"])
def get_history():
    # 1. 连数据库
    conn = sqlite3.connect('reviews.db')
    cursor = conn.cursor()

    cursor.execute('''
           CREATE TABLE IF NOT EXISTS history(
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           review TEXT,
           result TEXT
           )
       ''')

    # 2. 查库：按 id 倒序查，最新的在前面，只查最近 5 条
    cursor.execute("SELECT review, result FROM history ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()

    conn.close()

    # 3. 整理数据返回给前端
    history_data = []
    for row in rows:
        review_text = row[0]
        # 因为数据库里存的是 JSON 字符串，我们要把它转回 Python 字典
        result_dict = json.loads(row[1])
        history_data.append({
            "review": review_text,
            "result": result_dict
        })

    return jsonify(ok=True, history=history_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)