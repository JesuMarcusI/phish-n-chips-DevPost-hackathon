chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.url) {
    console.log("Checking URL:", changeInfo.url); // Log URL for debugging
    fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: changeInfo.url })
    })
    .then(response => response.json())
    .then(data => {
      console.log("API Response:", data); // Log API response
      let badgeText = "✔";
      let badgeColor = "green";
      if (data.prediction === "Phishing") {
        badgeText = "⚠️";
        badgeColor = "red";
	const warningUrl = chrome.runtime.getURL("warning.html") + "?originalUrl=" + encodeURIComponent(changeInfo.url);
	chrome.tabs.update(tabId, { url: warningUrl });
      }
      chrome.browserAction.setBadgeText({ text: badgeText, tabId: tabId });
      chrome.browserAction.setBadgeBackgroundColor({ color: badgeColor });
      // Send result to popup.js
      chrome.storage.local.set({ prediction: data.prediction });
    })
    .catch(error => {
      console.error("Error fetching API:", error);
      chrome.storage.local.set({ prediction: "Error" });
    });
  }
});