
import { useEffect, useState } from "react";
import axios from "axios";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ScatterChart,
  Scatter,
  ZAxis
} from "recharts";
import {
  Trophy,
  Zap,
  BrainCircuit,
  BarChart3
} from "lucide-react";


const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";


export default function Analytics() {
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedMetric, setSelectedMetric] = useState("iou");

  useEffect(() => {
    axios.get(`${API}/metrics`)
      .then(res => setMetrics(res.data.results || []))
      .catch(() => setError("Unable to load model metrics"))
      .finally(() => setLoading(false));
  }, []);

  const best = metrics.reduce(
    (winner, model) =>
      !winner || model.iou > winner.iou ? model : winner,
    null
  );

  const fastest = metrics.reduce(
    (winner, model) =>
      !winner || model.time_ms < winner.time_ms ? model : winner,
    null
  );

  const sorted = [...metrics].sort(
    (a, b) => b[selectedMetric] - a[selectedMetric]
  );

  if (loading) return <div className="panel">Loading analytics...</div>;
  if (error) return <div className="error">{error}</div>;

  return (
    <div className="analytics-page">
      <div className="analytics-summary">
        <div className="stat-card">
          <Trophy color="#38bdf8" />
          <p>Best Accuracy Model</p>
          <h2>{best?.model || "--"}</h2>
          <small>IoU: {best?.iou.toFixed(2)}%</small>
        </div>

        <div className="stat-card">
          <Zap color="#38bdf8" />
          <p>Fastest Model</p>
          <h2>{fastest?.model || "--"}</h2>
          <small>
            {fastest?.time_ms.toFixed(2)} ms/image
          </small>
        </div>

        <div className="stat-card">
          <BrainCircuit color="#38bdf8" />
          <p>Total Models</p>
          <h2>{metrics.length}</h2>
          <small>Evaluated on 58 test images</small>
        </div>
      </div>

      <div className="panel analytics-chart">
        <div className="chart-heading">
          <h3>Model Performance Comparison</h3>
          <select
            value={selectedMetric}
            onChange={e => setSelectedMetric(e.target.value)}
          >
            <option value="iou">IoU</option>
            <option value="f1">F1 Score</option>
            <option value="precision">Precision</option>
            <option value="recall">Recall</option>
          </select>
        </div>

        <div className="chart-scroll">
          <div style={{ minWidth: 1100, height: 450 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={sorted}
                margin={{ top: 15, right: 20, left: 0, bottom: 120 }}
              >
                <CartesianGrid stroke="#25334a" strokeDasharray="3 3" />
                <XAxis
                  dataKey="model"
                  angle={-50}
                  textAnchor="end"
                  interval={0}
                  height={130}
                  tick={{ fill: "#94a3b8", fontSize: 10 }}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fill: "#94a3b8" }}
                />
                <Tooltip
                  contentStyle={{
                    background: "#111b2d",
                    border: "1px solid #31526b",
                    color: "#fff"
                  }}
                />
                <Bar
                  dataKey={selectedMetric}
                  fill="#38bdf8"
                  radius={[5, 5, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="panel analytics-chart">
        <h3>Inference Speed vs Segmentation Quality</h3>
        <p>
          Models closer to the upper-left generally offer
          better IoU with lower inference time.
        </p>

        <div style={{ width: "100%", height: 350 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 25, bottom: 25, left: 10 }}>
              <CartesianGrid stroke="#25334a" />
              <XAxis
                type="number"
                dataKey="time_ms"
                name="Inference Time"
                unit="ms"
                tick={{ fill: "#94a3b8" }}
              />
              <YAxis
                type="number"
                dataKey="iou"
                name="IoU"
                unit="%"
                domain={[0, 100]}
                tick={{ fill: "#94a3b8" }}
              />
              <ZAxis type="number" range={[80, 80]} />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const item = payload[0].payload;
                  return (
                    <div className="chart-tooltip">
                      <strong>{item.model}</strong>
                      <p>IoU: {item.iou.toFixed(2)}%</p>
                      <p>Time: {item.time_ms.toFixed(2)} ms</p>
                    </div>
                  );
                }}
              />
              <Scatter data={metrics} fill="#ff2d95" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="panel">
        <h3>Complete Model Leaderboard</h3>
        <div className="table-scroll">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>IoU</th>
                <th>F1</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Time (ms)</th>
                <th>Size (KB)</th>
              </tr>
            </thead>
            <tbody>
              {[...metrics]
                .sort((a, b) => b.iou - a.iou)
                .map((model, index) => (
                  <tr key={model.model}>
                    <td>{index + 1}</td>
                    <td>{model.model}</td>
                    <td>{model.iou.toFixed(2)}%</td>
                    <td>{model.f1.toFixed(2)}%</td>
                    <td>{model.precision.toFixed(2)}%</td>
                    <td>{model.recall.toFixed(2)}%</td>
                    <td>{model.time_ms.toFixed(2)}</td>
                    <td>{model.size_kb.toFixed(2)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
