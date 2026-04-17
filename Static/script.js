function updateVal(id, value) {
    document.getElementById(id + "_val").innerText = value;
}

function step(id, change) {
    let val = Number(document.getElementById(id + "_val").innerText);
    val += change;
    if (val < 0) val = 0;
    document.getElementById(id + "_val").innerText = val;
}

async function predict() {
    const data = {
        total_orders: Number(document.getElementById("total_orders_val").innerText),
        total_spent: Number(document.getElementById("total_spent_val").innerText),
        total_returns: Number(document.getElementById("total_returns_val").innerText),
        return_ratio: Number(document.getElementById("return_ratio_val").innerText),
        customer_lifetime_days: Number(document.getElementById("customer_lifetime_days_val").innerText),
        avg_order_value: Number(document.getElementById("avg_order_value_val").innerText),
        purchase_frequency: Number(document.getElementById("purchase_frequency_val").innerText)
    };

    const res = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    });

    const result = await res.json();

    const box = document.getElementById("result");

    if (result.fraud_prediction === 1) {
        box.className = "result-card fraud";
        box.innerHTML = `<h2>⚠ Fraud</h2><h1>${result.risk_score}%</h1>`;
    } else {
        box.className = "result-card safe";
        box.innerHTML = `<h2>✅ Safe</h2><h1>${result.risk_score}%</h1>`;
    }
}