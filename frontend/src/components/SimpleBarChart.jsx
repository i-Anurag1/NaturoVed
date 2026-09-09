import React from "react";

/**
 * Minimal, dependency-free horizontal bar chart. We deliberately avoid
 * pulling in a full charting library (recharts/chart.js) for what is
 * fundamentally 3 simple distribution bar charts — keeps the frontend
 * bundle small and the code easy for a 4-person student team to read
 * and modify.
 */
export default function SimpleBarChart({ data, colorMap = {}, defaultColor = "#C77D2E" }) {
  if (!data || data.length === 0) {
    return <p className="empty-state">No data yet.</p>;
  }
  const max = Math.max(...data.map((d) => d.count), 1);

  return (
    <div className="simple-bar-chart">
      {data.map((item) => (
        <div className="bar-row" key={item.label}>
          <span className="bar-label">{item.label}</span>
          <div className="bar-track">
            <div
              className="bar-fill"
              style={{
                width: `${(item.count / max) * 100}%`,
                backgroundColor: colorMap[item.label] || defaultColor,
              }}
            />
          </div>
          <span className="bar-count">{item.count}</span>
        </div>
      ))}
    </div>
  );
}
