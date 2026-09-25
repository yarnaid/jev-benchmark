/**
 * Chart.js helpers: grouped bar chart of answer counts per option (one dataset per rater, in the rater's
 * palette color): argmax counts, or applied-label counts for a multi-label question. Thin bars with
 * 4px rounded ends, a small gap between neighbours, recessive grid lines, text in the page's own ink (so
 * the chart reads in both themes), a legend and per-bar hover tooltips.
 * Exports: countsChart, destroyCharts.
 */

const charts = new Set();

function ink() {
  const style = getComputedStyle(document.body);
  return { text: style.getPropertyValue("--bs-secondary-color").trim() || "#6c757d", grid: style.getPropertyValue("--bs-border-color-translucent").trim() || "rgba(0,0,0,0.1)" };
}

export function countsChart(canvas, question, labels, colors) {
  const { text, grid } = ink();
  const datasets = question.raters.map((stats) => ({
    label: labels[stats.rater] ?? stats.rater,
    data: question.options.map((option) => (stats.label_counts ?? stats.argmax_counts)[option] ?? 0),
    backgroundColor: colors.get(stats.rater),
    borderRadius: 4,
    categoryPercentage: 0.8,
    barPercentage: 0.9,
  }));
  const axis = { ticks: { color: text }, grid: { color: grid } };
  const chart = new Chart(canvas, {
    type: "bar",
    data: { labels: question.options, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12, color: text } } },
      scales: { x: { ...axis, ticks: { color: text, autoSkip: false, maxRotation: 60 }, grid: { display: false } }, y: { ...axis, beginAtZero: true, ticks: { color: text, precision: 0 } } },
    },
  });
  charts.add(chart);
  return chart;
}

export function destroyCharts() {
  for (const chart of charts) chart.destroy();
  charts.clear();
}
