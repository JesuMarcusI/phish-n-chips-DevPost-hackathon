chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.url) {
    fetch("http://30.110.57.28:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ url: changeInfo.url })
    })
    .then(response => response.json())
    .then(data => {
      if (data.prediction === "Phishing") {
        chrome.action.setBadgeText({ text: "⚠️", tabId: tabId });
        chrome.action.setBadgeBackgroundColor({ color: "red" });
      } else {
        chrome.action.setBadgeText({ text: "✔", tabId: tabId });
        chrome.action.setBadgeBackgroundColor({ color: "green" });
      }
    })
    .catch(error => console.error("Error:", error));
  }
});
