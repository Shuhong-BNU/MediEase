/*
 * Query 页面脚本。
 * 负责：
 * 1. 收集单轮问答所需的文本、图片和语音播报配置。
 * 2. 调用 `/api/agent/query`。
 * 3. 展示答案、引用来源、风险提示、建议动作、人工升级和语音结果。
 */

const queryForm = document.getElementById("queryForm");
const queryInput = document.getElementById("queryInput");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const speechVoiceSelect = document.getElementById("speechVoiceSelect");
const submitButton = document.getElementById("submitButton");
const statusBadge = document.getElementById("statusBadge");
const answerOutput = document.getElementById("answerOutput");
const audioVoiceMeta = document.getElementById("audioVoiceMeta");
const audioPlayer = document.getElementById("audioPlayer");
const sessionMeta = document.getElementById("sessionMeta");

const VOICE_LABELS = {
  longanyang: "龙安阳",
  longanhuan: "龙安欢",
  longxiaochun_v3: "龙小淳",
  longxiaoxia_v3: "龙小夏",
  longyumi_v3: "YUMI",
  longanwen_v3: "龙安雯",
  longanli_v3: "龙安莉",
  longanyun_v3: "龙安昀",
};

function isSpeechEnabled() {
  return speechVoiceSelect.value !== "none";
}

function setStatus(text, variant = "") {
  statusBadge.textContent = text;
  statusBadge.className = `status-badge${variant ? ` ${variant}` : ""}`;
}

function setBlockContent(element, content, placeholder = false) {
  element.textContent = content;
  element.classList.toggle("placeholder", placeholder);
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = typeof reader.result === "string" ? reader.result : "";
      const [, base64 = ""] = result.split(",");
      resolve(base64);
    };
    reader.onerror = () => reject(new Error("图片读取失败"));
    reader.readAsDataURL(file);
  });
}

function renderImagePreview(file) {
  if (!file) {
    imagePreview.classList.add("is-empty");
    imagePreview.innerHTML = "<span>未选择图片</span>";
    return;
  }

  const previewUrl = URL.createObjectURL(file);
  imagePreview.classList.remove("is-empty");
  imagePreview.innerHTML = `<img src="${previewUrl}" alt="预览图片" />`;
}

function setSessionMeta(sessionId = "") {
  if (!sessionId) {
    sessionMeta.textContent = "当前为单次请求模式，返回后会显示本次 session ID。";
    return;
  }
  sessionMeta.textContent = `本次会话 ID：${sessionId}`;
}

function renderResultText(data) {
  const lines = [data.answer || "接口未返回 answer。"];
  if (data.disclaimer) {
    lines.push(`免责声明：${data.disclaimer}`);
  }
  if (Array.isArray(data.risk_alerts) && data.risk_alerts.length > 0) {
    lines.push(
      "风险提示：\n" +
        data.risk_alerts
          .map((item) => `- [${item.risk_level}] ${item.message}`)
          .join("\n")
    );
  }
  if (Array.isArray(data.recommended_actions) && data.recommended_actions.length > 0) {
    lines.push("建议动作：\n" + data.recommended_actions.map((item) => `- ${item}`).join("\n"));
  }
  if (Array.isArray(data.citations) && data.citations.length > 0) {
    lines.push(
      "来源引用：\n" +
        data.citations
          .map((item) => `- [${item.source_type}] ${item.title}：${item.snippet || "无摘要"}`)
          .join("\n")
    );
  }
  if (data.manual_escalation) {
    lines.push(
      `人工升级：已生成事件 #${data.manual_escalation.id}，建议：${data.manual_escalation.recommended_action}`
    );
  }
  if (data.ocr_text) {
    lines.push(`OCR 提取文本：\n${data.ocr_text}`);
  }
  return lines.join("\n\n");
}

async function buildPayload() {
  const payload = {
    query: queryInput.value.trim(),
    images: [],
    debug_planner: false,
    enable_speech: isSpeechEnabled(),
    speech_voice: isSpeechEnabled() ? speechVoiceSelect.value : "longanyang",
    speech_format: "mp3",
  };

  const file = imageInput.files[0];
  if (file) {
    const imageBase64 = await fileToBase64(file);
    payload.images.push({
      image_base64: imageBase64,
      mime_type: file.type || "image/png",
    });
  }

  return payload;
}

async function submitQuery(event) {
  event.preventDefault();

  if (!queryInput.value.trim()) {
    setStatus("请输入问题", "error");
    return;
  }

  submitButton.disabled = true;
  setStatus("查询中", "loading");
  setBlockContent(answerOutput, "正在调用 /api/agent/query ...");
  audioVoiceMeta.classList.add("hidden");
  audioVoiceMeta.textContent = "";
  audioPlayer.classList.add("hidden");
  audioPlayer.removeAttribute("src");

  try {
    const payload = await buildPayload();
    const response = await fetch("/api/agent/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "请求失败");
    }

    setBlockContent(answerOutput, renderResultText(data));
    setSessionMeta(data.conversation_session_id || "");

    if (data.speech_download_url) {
      const voiceCode = data.speech_voice || speechVoiceSelect.value;
      const voiceLabel = VOICE_LABELS[voiceCode] || voiceCode;
      audioVoiceMeta.textContent = `当前声音：${voiceLabel}`;
      audioVoiceMeta.classList.remove("hidden");
      audioPlayer.src = data.speech_download_url;
      audioPlayer.classList.remove("hidden");
    }

    setStatus("查询完成", "success");
  } catch (error) {
    const message = error instanceof Error ? error.message : "请求失败";
    setBlockContent(answerOutput, message);
    setStatus("查询失败", "error");
  } finally {
    submitButton.disabled = false;
  }
}

queryForm.addEventListener("submit", submitQuery);
imageInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  renderImagePreview(file);
});

document.querySelectorAll(".prompt-chip").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.prompt || "";
    queryInput.focus();
  });
});

setSessionMeta("");
