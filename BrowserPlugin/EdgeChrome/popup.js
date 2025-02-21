document.addEventListener("DOMContentLoaded", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        fetch("http://30.110.57.28:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ url: tabs[0].url })
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById("status").innerText = `This site is: ${data.prediction}`;
        })
        .catch(error => {
            console.error("Error:", error);
            document.getElementById("status").innerText = "Error detecting phishing";
        });
    });
});
