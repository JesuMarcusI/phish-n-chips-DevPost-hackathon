document.getElementById('proceedButton').addEventListener('click', () => {
  // Retrieve the original URL from the query parameter
  const urlParams = new URLSearchParams(window.location.search);
  const originalUrl = urlParams.get('originalUrl');
  if (originalUrl) {
    window.location.href = originalUrl;
  }
});