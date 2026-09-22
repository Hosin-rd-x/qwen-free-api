"""
Embedded Persian (RTL) web dashboard — served at / and /dashboard.

Single self-contained HTML string: no CDN, no external assets, no build step.
Live data comes from /api/status (accounts, cooldowns, models). Includes an
agent-connection panel with copyable snippets and a live playground that calls
/v1/chat/completions directly from the browser.
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeepSeek Free API — داشبورد</title>
<style>
  :root{
    --bg:#0b1220; --panel:#111a2e; --panel2:#0e1626; --line:#1e2a44;
    --txt:#e8eefc; --dim:#93a4c3; --acc:#4f8cff; --ok:#34d399; --warn:#fbbf24;
    --err:#f87171; --mono:'SFMono-Regular',Consolas,'Courier New',monospace;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--txt);font-family:Tahoma,'Segoe UI',sans-serif;
       min-height:100vh;padding:24px 16px}
  .wrap{max-width:1080px;margin:0 auto}
  header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:22px}
  .logo{width:46px;height:46px;border-radius:12px;
        background:linear-gradient(135deg,#1d4ed8,#4f8cff);display:flex;align-items:center;
        justify-content:center;font-weight:bold;font-size:22px;color:#fff}
  h1{font-size:20px;font-weight:bold}
  .sub{color:var(--dim);font-size:12.5px;margin-top:3px}
  .pill{display:inline-flex;align-items:center;gap:7px;padding:7px 14px;border-radius:999px;
        background:var(--panel);border:1px solid var(--line);font-size:12.5px;margin-inline-start:auto}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--dim)}
  .dot.ok{background:var(--ok);box-shadow:0 0 8px var(--ok)}
  .dot.warn{background:var(--warn);box-shadow:0 0 8px var(--warn)}
  .dot.err{background:var(--err)}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:18px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}
  .card .n{font-size:26px;font-weight:bold;font-family:var(--mono)}
  .card .t{color:var(--dim);font-size:12px;margin-top:5px}
  .ok{color:var(--ok)} .warn{color:var(--warn)} .err{color:var(--err)} .dim{color:var(--dim)}
  section{background:var(--panel);border:1px solid var(--line);border-radius:16px;
          padding:18px;margin-bottom:16px}
  h2{font-size:14.5px;margin-bottom:12px;display:flex;align-items:center;gap:8px}
  h2 .en{color:var(--dim);font-size:11px;font-weight:normal;font-family:var(--mono)}
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  th{color:var(--dim);text-align:right;padding:8px 10px;border-bottom:1px solid var(--line);font-weight:normal}
  td{padding:9px 10px;border-bottom:1px solid var(--line)}
  tr:last-child td{border-bottom:none}
  .tag{padding:3px 10px;border-radius:999px;font-size:11px;font-family:var(--mono)}
  .tag.active{background:rgba(52,211,153,.12);color:var(--ok)}
  .tag.cooldown{background:rgba(251,191,36,.12);color:var(--warn)}
  .tag.disabled{background:rgba(248,113,113,.12);color:var(--err)}
  .tag.none{background:rgba(147,164,195,.12);color:var(--dim)}
  code,.code{font-family:var(--mono);font-size:12px;direction:ltr}
  .field{display:flex;align-items:center;gap:10px;background:var(--panel2);
         border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin-bottom:10px}
  .field .k{color:var(--dim);font-size:11.5px;min-width:70px}
  .field code{flex:1;color:var(--txt);overflow-x:auto;white-space:nowrap}
  .btn{background:var(--acc);border:none;color:#fff;padding:8px 14px;border-radius:8px;
       cursor:pointer;font-family:inherit;font-size:12.5px}
  .btn:hover{filter:brightness(1.12)}
  .btn.ghost{background:transparent;border:1px solid var(--line);color:var(--dim)}
  pre{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:38px 14px 14px;
      overflow-x:auto;direction:ltr;text-align:left;font-family:var(--mono);font-size:12px;
      line-height:1.65;color:#c9d7f2;position:relative}
  .copy{position:absolute;top:8px;left:8px;background:var(--panel);border:1px solid var(--line);
        color:var(--dim);border-radius:6px;padding:4px 9px;cursor:pointer;font-size:11px;font-family:inherit}
  .copy:hover{color:var(--txt)}
  textarea{width:100%;background:var(--panel2);border:1px solid var(--line);border-radius:10px;
           color:var(--txt);padding:12px;font-family:inherit;font-size:13px;resize:vertical;min-height:80px}
  textarea:focus{outline:1px solid var(--acc)}
  select{background:var(--panel2);border:1px solid var(--line);color:var(--txt);
         padding:8px 12px;border-radius:8px;font-family:inherit;font-size:12.5px}
  .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
  #out{margin-top:12px;display:none}
  .empty{color:var(--dim);font-size:13px;line-height:2}
  .empty b{color:var(--warn)}
  .steps{font-size:13px;line-height:2.1;color:var(--dim)}
  .steps code{background:var(--panel2);border:1px solid var(--line);border-radius:6px;
              padding:2px 8px;color:var(--acc)}
  .foot{text-align:center;color:var(--dim);font-size:11.5px;padding:14px 0 4px}
  .spin{display:inline-block;width:14px;height:14px;border:2px solid var(--line);
        border-top-color:var(--acc);border-radius:50%;animation:sp 1s linear infinite;vertical-align:-2px}
  @keyframes sp{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">DS</div>
    <div>
      <h1>DeepSeek Free API</h1>
      <div class="sub">پل OpenAI-سازگار برای chat.deepseek.com — بدون کلید API، بدون هزینه</div>
    </div>
    <div class="pill"><span class="dot" id="dot"></span><span id="pillTxt">در حال اتصال…</span></div>
  </header>

  <div class="grid">
    <div class="card"><div class="n" id="stAcc">—</div><div class="t">اکانت‌های ثبت‌شده</div></div>
    <div class="card"><div class="n ok" id="stOk">—</div><div class="t">آماده سرویس</div></div>
    <div class="card"><div class="n warn" id="stCd">—</div><div class="t">در حالت انتظار</div></div>
    <div class="card"><div class="n" id="stModel">—</div><div class="t">مدل‌ها</div></div>
  </div>

  <section>
    <h2>اکانت‌ها <span class="en">account pool — round-robin + failover</span></h2>
    <div id="accTable"></div>
  </section>

  <section>
    <h2>راه‌اندازی <span class="en">setup</span></h2>
    <div id="setupBox"></div>
  </section>

  <section>
    <h2>اتصال عامل / برنامه‌ها <span class="en">agent connection</span></h2>
    <div class="field"><span class="k">Base URL</span><code id="baseCopy">http://localhost:8000/v1</code>
      <button class="btn ghost" onclick="cp('baseCopy',this)">کپی</button></div>
    <div class="field"><span class="k">API Key</span><code>anything — بررسی نمی‌شود</code></div>
    <div class="field"><span class="k">مدل‌ها</span><code>deepseek-chat · deepseek-expert</code></div>
    <pre id="snip1"><button class="copy" onclick="cp('snip1',this)">کپی</button># Python — OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
r = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "سلام!"}],
)
print(r.choices[0].message.content)</pre>
    <pre id="snip2"><button class="copy" onclick="cp('snip2',this)">کپی</button># Tool-calling برای عامل‌ها (Hermes و …)
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "آب‌وهوای یک شهر",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}}, "required": ["city"]}}}]

r = client.chat.completions.create(
    model="deepseek-chat", messages=msgs, tools=tools)
tc = r.choices[0].message.tool_calls[0]
print(tc.function.name, tc.function.arguments)</pre>
    <pre id="snip3"><button class="copy" onclick="cp('snip3',this)">کپی</button># curl
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model":"deepseek-chat",
       "messages":[{"role":"user","content":"سلام"}]}'</pre>
  </section>

  <section>
    <h2>تست زنده <span class="en">playground</span></h2>
    <div class="row">
      <select id="model">
        <option value="deepseek-chat">deepseek-chat (سریع)</option>
        <option value="deepseek-expert">deepseek-expert (قوی‌تر)</option>
      </select>
      <label style="font-size:12.5px;color:var(--dim);display:flex;gap:6px;align-items:center">
        <input type="checkbox" id="think"> DeepThink</label>
      <button class="btn" onclick="send()" id="sendBtn">ارسال</button>
    </div>
    <textarea id="prompt" placeholder="پیام خود را بنویسید…">سلام! خودت رو معرفی کن.</textarea>
    <div id="out"><pre id="outPre"></pre></div>
  </section>

  <div class="foot">DeepSeek Free API · سازگار با OpenAI · بدون کلید رسمی · مصرف شخصی</div>
</div>

<script>
const $=id=>document.getElementById(id);
function cp(id,btn){const t=$(id).innerText;navigator.clipboard.writeText(t).then(()=>{
  const o=btn.innerText;btn.innerText='کپی شد ✓';setTimeout(()=>btn.innerText=o,1200);});}
async function refresh(){
  try{
    const s=await (await fetch('/api/status')).json();
    const pool=s.pool||{accounts:[],total:0,healthy:0};
    const ok=pool.healthy||0, tot=pool.total||0;
    $('stAcc').textContent=tot; $('stOk').textContent=ok;
    const cd=(pool.accounts||[]).filter(a=>a.state==='cooldown').length;
    $('stCd').textContent=cd; $('stModel').textContent=(s.models||[]).length;
    const dot=$('dot'), pt=$('pillTxt');
    if(tot===0){dot.className='dot err';pt.textContent='اکانتی ثبت نشده';}
    else if(ok>0){dot.className='dot ok';pt.textContent='فعال — '+ok+' اکانت آماده';}
    else{dot.className='dot warn';pt.textContent='همه اکانت‌ها مشغول/محدود';}
    const tbl=$('accTable');
    if(!tot){tbl.innerHTML='<div class="empty">هیچ اکانتی یافت نشد. برای شروع: <code>python -m deepseek.auth --account main</code></div>';}
    else{
      let h='<table><tr><th>نام</th><th>منبع</th><th>وضعیت</th><th>درخواست</th><th>خطا</th><th>آخرین وضعیت</th></tr>';
      for(const a of pool.accounts){
        const tag=a.state==='active'?'active':a.state==='cooldown'?'cooldown':'disabled';
        const fa=a.state==='active'?'فعال':a.state==='cooldown'?('انتظار '+a.cooldown_s+'s'):'غیرفعال';
        h+=`<tr><td class="code">${a.name}</td><td>${a.source==='token'?'توکن':'مرورگر'}</td>
            <td><span class="tag ${tag}">${fa}</span></td>
            <td class="code">${a.requests}</td><td class="code">${a.errors}</td>
            <td class="dim" style="font-size:11.5px">${a.last_error||'—'}</td></tr>`;
      }
      tbl.innerHTML=h+'</table>';
    }
    $('setupBox').innerHTML = tot
      ? '<div class="steps">همه‌چیز آماده است. نقطه اتصال برای هر ابزار OpenAI-سازگار: <code>http://localhost:8000/v1</code></div>'
      : '<div class="steps">۱. ابتدا یک بار در مرورگر وارد حساب DeepSeek شوید: <code>python -m deepseek.auth --account main</code><br>۲. برای اکانت بیشتر همین دستور را با نام دیگر بزنید: <code>python -m deepseek.auth --account work</code><br>۳. سرور را ری‌استارت نکنید — وضعیت هر ۵ ثانیه تازه می‌شود.</div>';
  }catch(e){ $('dot').className='dot err'; $('pillTxt').textContent='خطای اتصال به سرور'; }
}
async function send(){
  const btn=$('sendBtn');btn.innerHTML='<span class="spin"></span> در حال پاسخ…';btn.disabled=true;
  $('out').style.display='block';$('outPre').textContent='…';
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:$('model').value,stream:false,
        thinking:$('think').checked,
        messages:[{role:'user',content:$('prompt').value}]})});
    const j=await r.json();
    if(j.error){$('outPre').textContent='⚠ '+j.error.message;}
    else{$('outPre').textContent=j.choices[0].message.content||'(خالی)';}
  }catch(e){$('outPre').textContent='⚠ '+e.message;}
  btn.textContent='ارسال';btn.disabled=false;
}
refresh(); setInterval(refresh,5000);
</script>
</body>
</html>"""
