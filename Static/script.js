async function predict() {
    const resultDiv = document.getElementById("result");

    // Loading state
    resultDiv.className = "result-box loading";
    resultDiv.innerHTML = "Analyzing... ⏳";

    const data = {
        total_orders: Number(document.getElementById("total_orders").value),
        total_spent: Number(document.getElementById("total_spent").value),
        total_returns: Number(document.getElementById("total_returns").value),
        return_ratio: Number(document.getElementById("return_ratio").value),
        customer_lifetime_days: Number(document.getElementById("customer_lifetime_days").value),
        avg_order_value: Number(document.getElementById("avg_order_value").value),
        purchase_frequency: Number(document.getElementById("purchase_frequency").value)
    };

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.fraud_prediction === 1) {
            resultDiv.className = "result-box fraud";
            resultDiv.innerHTML = `⚠️ Fraud Detected <br> Risk Score: ${result.risk_score}`;
        } else {
            resultDiv.className = "result-box safe";
            resultDiv.innerHTML = `✅ Safe Customer <br> Risk Score: ${result.risk_score}`;
        }

    } catch (error) {
        resultDiv.className = "result-box fraud";
        resultDiv.innerHTML = "Error connecting to server ❌";
    }
}