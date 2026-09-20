/* Shared by the control page and fullscreen calibration. */
let language = "zh";
try {
  const saved = localStorage.getItem("opengazelink.language");
  language = saved === "en" || saved === "zh" ? saved : (navigator.language.startsWith("zh") ? "zh" : "en");
} catch (_) { /* Private browsing can disable storage. */ }

function tr(zh, en) { return language === "en" ? en : zh; }

function applyLanguage(next) {
  language = next === "en" ? "en" : "zh";
  document.documentElement.lang = language === "en" ? "en" : "zh-CN";
  document.querySelectorAll("[data-zh][data-en]").forEach(element => {
    element.textContent = element.dataset[language];
  });
  document.getElementById("language").value = language;
  try { localStorage.setItem("opengazelink.language", language); } catch (_) {}
  document.dispatchEvent(new CustomEvent("languagechange"));
}
