document.getElementById('search-form').addEventListener('submit', function (event) {
    event.preventDefault();
    let query = document.getElementById('query').value;
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';
    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'query': query
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultsDiv.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
        } else {
            displayResults(data);
            displayChart(data);
        }
    });
});

function displayResults(data) {
    let resultsDiv = document.getElementById('results');
    data.documents.forEach((doc, index) => {
        let docDiv = document.createElement('div');
        
        let title = document.createElement('h3');
        title.textContent = `Document ${data.indices[index]}`;  
        
        let similarity = document.createElement('p');
        similarity.innerHTML = `Similarity Score: <b>${data.similarities[index]}</b>`;
        
        let content = document.createElement('p');
        content.textContent = doc.length > 500 ? doc.substring(0, 500) + '...' : doc;
        
        docDiv.appendChild(title);
        docDiv.appendChild(similarity);
        docDiv.appendChild(content);
        resultsDiv.appendChild(docDiv);
    });
}

let similarityChart;

function displayChart(data) {
    let ctx = document.getElementById('similarityChart').getContext('2d');
    if (similarityChart) {
        similarityChart.destroy();
    }
    similarityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.indices.map(index => `Doc ${index}`),  
            datasets: [{
                label: 'Cosine Similarity',
                data: data.similarities,
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}