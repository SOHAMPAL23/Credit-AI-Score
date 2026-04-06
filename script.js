document.addEventListener('DOMContentLoaded', () => {
    // Range Slider Elements
    const ageInput = document.getElementById('age');
    const ageVal = document.getElementById('age-val');
    
    const yearsInput = document.getElementById('years-employed');
    const yearsVal = document.getElementById('years-val');
    
    const creditInput = document.getElementById('credit-score');
    const creditVal = document.getElementById('credit-val');
    
    const dtiInput = document.getElementById('dti');
    const dtiVal = document.getElementById('dti-val');

    // Update values on slide
    ageInput.addEventListener('input', (e) => ageVal.textContent = e.target.value);
    yearsInput.addEventListener('input', (e) => yearsVal.textContent = e.target.value);
    creditInput.addEventListener('input', (e) => creditVal.textContent = e.target.value);
    dtiInput.addEventListener('input', (e) => {
        dtiVal.textContent = Number(e.target.value).toFixed(2);
    });

    // Form submission
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const btnText = predictBtn.querySelector('.btn-text');
    const btnLoader = document.getElementById('btn-loader');
    
    const initialState = document.getElementById('initial-state');
    const dashboard = document.getElementById('results-dashboard');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loader
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');
        predictBtn.disabled = true;

        // Gather data
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
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to get prediction');
            }

            const data = await response.json();
            updateDashboard(data, requestData);
            
            // Switch views
            initialState.classList.add('hidden');
            dashboard.classList.remove('hidden');

        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            // Hide loader
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
            predictBtn.disabled = false;
        }
    });

    function updateDashboard(data, requestData) {
        const { prediction, probability } = data;
        const den_prob = probability[0];
        const app_prob = probability[1];

        // Decision Card
        const decisionCard = document.getElementById('decision-card');
        const decisionText = document.getElementById('decision-text');
        
        if (prediction === 1) {
            decisionText.textContent = "✅ APPROVED";
            decisionCard.classList.remove('decision-denied');
            decisionCard.classList.add('decision-approved');
        } else {
            decisionText.textContent = "❌ DENIED";
            decisionCard.classList.remove('decision-approved');
            decisionCard.classList.add('decision-denied');
        }

        // Progress Rings
        updateRing('approval', app_prob);
        updateRing('denial', den_prob);

        // Recommendations
        updateRecommendations(prediction, requestData);
    }

    function updateRing(type, value) {
        const circle = document.getElementById(`${type}-circle`);
        const text = document.getElementById(`${type}-percentage`);
        
        const circumference = 2 * Math.PI * 45;
        const offset = circumference - (value * circumference);
        
        // Slight delay for animation effect
        setTimeout(() => {
            circle.style.strokeDashoffset = offset;
            text.textContent = `${(value * 100).toFixed(1)}%`;
        }, 100);
    }

    function updateRecommendations(prediction, requestData) {
        const recIcon = document.getElementById('rec-icon');
        const recTitle = document.getElementById('rec-title');
        const recContent = document.getElementById('rec-content');

        if (prediction === 1) {
            recIcon.textContent = "✅";
            recTitle.textContent = "Excellent News!";
            recTitle.style.color = "var(--success)";
            recContent.innerHTML = `
                <p>This application is highly likely to be approved based on the provided information.</p>
                <ul>
                    <li>Proceed with standard verification procedures</li>
                    <li>Collect all required documentation</li>
                    <li>Process loan terms and conditions</li>
                </ul>
            `;
        } else {
            recIcon.textContent = "⚠️";
            recTitle.textContent = "Application Concerns";
            recTitle.style.color = "var(--danger)";
            
            recContent.innerHTML = `
                <p>Based on the applicant's profile, the loan does not currently meet the criteria for approval.</p>
                <p><strong>Areas for improvement:</strong></p>
                <ul>
                    <li>Higher credit scores are favorably weighted.</li>
                    <li>Reducing the debt-to-income ratio increases approval chances.</li>
                    <li>Consider requesting a lower loan amount relative to income.</li>
                </ul>
            `;
        }
    }
});
