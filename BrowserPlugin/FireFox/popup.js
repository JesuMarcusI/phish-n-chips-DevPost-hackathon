document.addEventListener("DOMContentLoaded", () => {
    chrome.storage.local.get("prediction", (data) => {
        const statusElement = document.getElementById("status");
        if (data.prediction) {
            statusElement.innerText = `This site is: ${data.prediction}`;
        } else {
            statusElement.innerText = "No data received";
        }
    });
});