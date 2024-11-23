document.addEventListener('DOMContentLoaded', () => {
    const summarizeBtn = document.getElementById('summarizeBtn');
    const inputText = document.getElementById('inputText');
    const outputSummary = document.getElementById('outputSummary');

    summarizeBtn.addEventListener('click', async () => {
        const text = inputText.value.trim();
        if (!text) {
            outputSummary.innerText = "Please enter some text to summarize!";
            return;
        }

        outputSummary.innerText = "Summarizing...";
        try {
            const response = await fetch('/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text }),
            });

            const data = await response.json();
            if (response.ok) {
                outputSummary.innerText = data.summary;
            } else {
                outputSummary.innerText = data.error || "Something went wrong!";
            }
        } catch (error) {
            outputSummary.innerText = "An error occurred. Please try again.";
        }
    });
});
