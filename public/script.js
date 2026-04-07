document.addEventListener('DOMContentLoaded', () => {
    // Elegant number formatter
    const formatCurrency = (num) => new Intl.NumberFormat('en-US').format(num);
    const formatPercent = (num) => Math.round(num * 100);

    // Inputs & Badges
    const ageInput = document.getElementById('age');
    const ageVal = document.getElementById('age-val');
    
    const yearsInput = document.getElementById('years-employed');
    const yearsVal = document.getElementById('years-val');
    
    const creditInput = document.getElementById('credit-score');
    const creditVal = document.getElementById('credit-val');
    
    const dtiInput = document.getElementById('dti');
    const dtiVal = document.getElementById('dti-val');

    // Real-time UI updates for sliders
    ageInput.addEventListener('input', (e) => ageVal.textContent = `${e.target.value} yrs`);
    yearsInput.addEventListener('input', (e) => yearsVal.textContent = `${e.target.value} yrs`);
    
    creditInput.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        creditVal.textContent = val;
        // Dynamically style credit badge
        if(val >= 750) creditVal.className = 'val-display badge badge-gold';
        else if (val >= 650) creditVal.className = 'val-display badge';
        else {
            creditVal.className = 'val-display badge';
            creditVal.style.color = 'var(--danger)';
            creditVal.style.borderColor = 'rgba(239, 68, 68, 0.3)';
            creditVal.style.background = 'rgba(239, 68, 68, 0.1)';
        }
    });

    dtiInput.addEventListener('input', (e) => dtiVal.textContent = `${formatPercent(e.target.value)}%`);

    // Form logic
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const btnText = predictBtn.querySelector('.btn-text');
    const btnLoader = document.getElementById('btn-loader');
    
    const initialState = document.getElementById('initial-state');
    const dashboard = document.getElementById('results-dashboard');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Button loading state
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');
        predictBtn.disabled = true;

        const requestData = {
            age: parseInt(ageInput.value),
            income: parseFloat(document.getElementById('income').value),
            years_employed: parseInt(yearsInput.value),
            loan_amount: parseFloat(document.getElementById('loan-amount').value),
            credit_score: parseInt(creditInput.value),
            loan_purpose: document.getElementById('loan-purpose').value,
            debt_to_income: parseFloat(dtiInput.value)
        };

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Prediction request failed');
            }

            const data = await response.json();
            
            // Switch views gracefully
            initialState.style.display = 'none';
            dashboard.classList.remove('hidden');
            
            // Execute animation render
            renderDashboard(data, requestData);

        } catch (error) {
            console.error('Error fetching prediction:', error);
            alert('Simulation Error: ' + error.message);
        } finally {
            // Restore button
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
            predictBtn.disabled = false;
        }
    });

    function renderDashboard(data, requestData) {
        const { prediction, probability } = data;
        const denRisk = probability[0] * 100;
        const appConf = probability[1] * 100;

        // Animate Decision Text
        const decisionCard = document.getElementById('decision-card');
        const decisionText = document.getElementById('decision-text');
        
        if (prediction === 1) {
            decisionText.textContent = "APPROVED";
            decisionCard.className = 'glass-panel major-decision decision-approved';
        } else {
            decisionText.textContent = "DENIED";
            decisionCard.className = 'glass-panel major-decision decision-denied';
        }

        // Animate SVG Gauges and Numbers
        animateGauge('approval', appConf);
        animateGauge('denial', denRisk);

        // Render AI Insights
        renderInsights(prediction, requestData);
    }

    function animateGauge(prefix, targetValue) {
        const circle = document.getElementById(`${prefix}-gauge`);
        const valueBox = document.getElementById(`${prefix}-value`);
        
        // Math for SVG stroke offset (r=50 -> circum=314)
        const circumference = 314;
        const offset = circumference - ((targetValue / 100) * circumference);
        
        // Animate SVG
        setTimeout(() => {
            circle.style.strokeDashoffset = offset;
        }, 100);

        // Animate Number Counting
        let current = 0;
        const increment = targetValue / 40; // 40 frames
        const timer = setInterval(() => {
            current += increment;
            if (current >= targetValue) {
                current = targetValue;
                clearInterval(timer);
            }
            valueBox.innerHTML = `${Math.round(current)}<span class="percent">%</span>`;
        }, 20);
    }

    function renderInsights(prediction, requestData) {
        const content = document.getElementById('rec-content');
        const dtiPercent = formatPercent(requestData.debt_to_income);
        
        if (prediction === 1) {
            content.innerHTML = `
                <p>The neural network has evaluated the financial parameters and determined high viability for funding.</p>
                <ul>
                    <li><strong>Credit Factor:</strong> Your score of ${requestData.credit_score} strongly supports probability.</li>
                    <li><strong>DTI Ratio:</strong> At ${dtiPercent}%, your debt-to-income is well within safe thresholds.</li>
                    <li><strong>Next Phase:</strong> Advance to identity verification and document collection.</li>
                </ul>
            `;
        } else {
            content.innerHTML = `
                <p>The neural model detected risk factors that exceed our current approval thresholds. Review the following adjustments:</p>
                <ul>
                    ${requestData.credit_score < 700 ? `<li><strong>Credit Profile:</strong> Increasing score above 700 yields a +25% confidence boost.</li>` : ''}
                    ${requestData.debt_to_income > 0.35 ? `<li><strong>DTI Limitation:</strong> A ${dtiPercent}% ratio triggers high-risk alerts. Target < 30%.</li>` : ''}
                    ${requestData.loan_amount / requestData.income > 3 ? `<li><strong>Leverage:</strong> Loan amount is over 3x annual income, which severely restricts approval.</li>` : ''}
                    <li><strong>Manual Review:</strong> Consider submitting mitigating documents to an underwriter.</li>
                </ul>
            `;
        }
    }
});
