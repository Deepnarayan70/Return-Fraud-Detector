async function predict() {
  const resultDiv = document.getElementById("result");

  resultDiv.innerHTML = "<h2>Prediction Result</h2><div class='result-body'>Analyzing...</div>";

  const data = {
    total_orders: Number(document.getElementById("total_orders").value),
    total_spent: Number(document.getElementById("total_spent").value),
    total_returns: Number(document.getElementById("total_returns").value),
    return_ratio: Number(document.getElementById("return_ratio").value),
    customer_lifetime_days: Number(document.getElementById("customer_lifetime_days").value),
    avg_order_value: Number(document.getElementById("avg_order_value").value),
    purchase_frequency: Number(document.getElementById("purchase_frequency").value)
  };

  const res = await fetch("/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data)
  });

  const result = await res.json();

  if (result.fraud_prediction === 1) {
    resultDiv.innerHTML = `
      <h2 class="fraud">⚠ Fraud Detected</h2>
      <div class="result-body">
        <div class="score">${result.risk_score}%</div>
        <p>High Risk Customer</p>
      </div>`;
  } else {
    resultDiv.innerHTML = `
      <h2 class="safe">✅ Safe Customer</h2>
      <div class="result-body">
        <div class="score">${result.risk_score}%</div>
        <p>Low Risk Customer</p>
      </div>`;
  }
}