function initScoreChart(canvasId, scoreData) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  new Chart(canvas.getContext('2d'), {
    type: 'doughnut',
    data: {
      labels: ['Text Similarity', 'Skill Match', 'ATS Score'],
      datasets: [{
        data: [
          scoreData.tfidf || 0,
          scoreData.skill || 0,
          scoreData.ats   || 0,
        ],
        backgroundColor: ['#6c63ff', '#22c87a', '#f59e0b'],
        borderWidth: 0,
        hoverOffset: 4,
      }]
    },
    options: {
      cutout: '72%',
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },   // we use our own legend in HTML
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.parsed}%`
          }
        }
      }
    }
  });
}