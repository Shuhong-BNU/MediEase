/*
 * 报告解读页脚本。
 * 负责：
 * 1. 收集报告文本、标题、类型和可选图片。
 * 2. 如有图片，先走 `/api/agent/query` 触发 OCR，再调用 `/api/reports/interpret`。
 * 3. 展示摘要、异常项、风险等级和免责声明。
 */

const reportForm = document.getElementById("reportForm");
const reportTextInput = document.getElementById("reportTextInput");
const reportTitleInput = document.getElementById("reportTitleInput");
const reportTypeInput = document.getElementById("reportTypeInput");
const reportImageInput = document.getElementById("reportImageInput");
const reportImagePreview = document.getElementById("reportImagePreview");
const reportStatusBadge = document.getElementById("reportStatusBadge");
const reportOutput = document.getElementById("reportOutput");
const submitReportButton = document.getElementById("submitReportButton");

function setStatus(text, variant = "") {
  reportStatusBadge.textContent = text;
  reportStatusBadge.className = `status-badge${variant ? ` ${variant}` : ""}`;
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
    reportImagePreview.classList.add("is-empty");
    reportImagePreview.innerHTML = "<span>未选择图片</span>";
    return;
  }

  const previewUrl = URL.createObjectURL(file);
  reportImagePreview.classList.remove("is-empty");
  reportImagePreview.innerHTML = `<img src="${previewUrl}" alt="报告图片预览" />`;
}

async function extractOcrText(file) {
  if (!file) {
    return "";
  }
  const imageBase64 = await fileToBase64(file);
  const response = await fetch("/api/agent/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: "请只提取图片里的医学报告文字，不要解释。",
      images: [
        {
          image_base64: imageBase64,
          mime_type: file.type || "image/png",
        },
      ],
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "OCR 提取失败");
  }
  return data.ocr_text || "";
}

function renderInterpretation(data) {
  const lines = [
    `摘要：${data.summary}`,
    `风险等级：${data.risk_level}`,
  ];
  if (Array.isArray(data.abnormal_items) && data.abnormal_items.length > 0) {
    lines.push(
      "异常项：\n" +
        data.abnormal_items
          .map((item) => `- ${item.label}${item.value ? ` (${item.value})` : ""}：${item.explanation}`)
          .join("\n")
    );
  }
  if (Array.isArray(data.recommended_actions) && data.recommended_actions.length > 0) {
    lines.push("建议动作：\n" + data.recommended_actions.map((item) => `- ${item}`).join("\n"));
  }
  if (data.extracted_text) {
    lines.push(`OCR 文本：\n${data.extracted_text}`);
  }
  if (data.disclaimer) {
    lines.push(`免责声明：${data.disclaimer}`);
  }
  return lines.join("\n\n");
}

async function submitReport(event) {
  event.preventDefault();

  const reportText = reportTextInput.value.trim();
  const file = reportImageInput.files[0];
  if (!reportText && !file) {
    setStatus("请先输入报告文本或上传图片", "error");
    return;
  }

  submitReportButton.disabled = true;
  setStatus("解读中", "loading");
  setBlockContent(reportOutput, "正在提取并解读报告 ...");

  try {
    const imageText = await extractOcrText(file);
    const response = await fetch("/api/reports/interpret", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        report_text: reportText,
        title: reportTitleInput.value.trim() || undefined,
        report_type: reportTypeInput.value.trim() || undefined,
        image_text: imageText || undefined,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "报告解读失败");
    }
    setBlockContent(reportOutput, renderInterpretation(data));
    setStatus("解读完成", "success");
  } catch (error) {
    const message = error instanceof Error ? error.message : "报告解读失败";
    setBlockContent(reportOutput, message);
    setStatus("解读失败", "error");
  } finally {
    submitReportButton.disabled = false;
  }
}

reportForm.addEventListener("submit", submitReport);
reportImageInput.addEventListener("change", (event) => {
  renderImagePreview(event.target.files[0]);
});
document.querySelectorAll(".prompt-chip").forEach((button) => {
  button.addEventListener("click", () => {
    reportTextInput.value = button.dataset.prompt || "";
    reportTextInput.focus();
  });
});
