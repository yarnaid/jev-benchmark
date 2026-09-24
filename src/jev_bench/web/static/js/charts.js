/**
 * Chart.js helpers: grouped bar chart of argmax counts per option (one dataset per rater).
 * Exports: countsChart, destroyCharts.
 */

const PALETTE = ["#6f42c1", "#d63384", "#0d6efd", "#20c997", "#fd7e14", "#6c757d", "#198754", "#dc3545"];
const charts = new Set();

export function countsChart(canvas, question, labels) {
  const datasets = question.raters.map((stats, index) => ({
    label: labels[stats.rater] ?? stats.rater,
    data: question.options.map((option) => stats.argmax_counts[option] ?? 0),
    backgroundColor: PALETTE[index % PALETTE.length],
    borderRadius: 3,
  }));
  const chart = new Chart(canvas, {
    type: "bar",
    data: { labels: question.options, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } },
      scales: { x: { ticks: { autoSkip: false, maxRotation: 60 } }, y: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });
  charts.add(chart);
  return chart;
}

export function destroyCharts() {
  for (const chart of charts) chart.destroy();
  charts.clear();
}
