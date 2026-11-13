
import { useEffect, useState } from "react";
import axios from "axios";
import {
  LayoutDashboard,
  Waves,
  BrainCircuit,
  BarChart3,
  History,
  UploadCloud,
  Play,
  Download,
  Clock3,
  Activity,
  CheckCircle2,
  AlertCircle
} from "lucide-react";
import "./index.css";
import Analytics from "./components/Analytics";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
export default function App() {
  const [page, setPage] = useState("dashboard");
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("RF-T");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [floodPercentage, setFloodPercentage] = useState(null);
  const [inferenceTime, setInferenceTime] = useState(null);
  const [predictionCount, setPredictionCount] = useState(0);

  useEffect(() => {
    axios.get(`${API}/models`)
      .then(res => {
        const names = res.data.models || [];
        setModels(names);
        if (names.length && !names.includes("RF-T")) {
          setSelectedModel(names[0]);
        }
      })
      .catch(() => {
        setError("Cannot connect to FastAPI. Check that the backend is running.");
      });
  }, []);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  useEffect(() => {
    return () => {
      if (result) URL.revokeObjectURL(result);
    };
  }, [result]);

  const handleFile = (event) => {
    const selected = event.target.files?.[0];
    if (!selected) return;

    if (!["image/jpeg", "image/png", "image/webp"].includes(selected.type)) {
      setError("Please select a PNG, JPEG or WebP image.");
      return;
    }

    if (selected.size > 10 * 1024 * 1024) {
      setError("Image must be smaller than 10 MB.");
      return;
    }

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult("");
    setFloodPercentage(null);
    setInferenceTime(null);
    setError("");
  };

  const predictFlood = async () => {
    if (!file) {
      setError("Please upload an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(
        `${API}/predict`,
        formData,
        {
          params: { model_name: selectedModel },
          responseType: "blob"
        }
      );

      const imageUrl = URL.createObjectURL(response.data);
      setResult(imageUrl);

      const percentage = Number(
        response.headers["x-flood-percentage"]
      );

      const time = Number(
        response.headers["x-inference-time-ms"]
      );

      setFloodPercentage(
        Number.isFinite(percentage) ? percentage : null
      );

      setInferenceTime(
        Number.isFinite(time) ? time : null
      );

      setPredictionCount(count => count + 1);

    } catch (err) {
      let message = "Prediction failed. Check the backend terminal.";

      if (err.response?.data instanceof Blob) {
        try {
          const text = await err.response.data.text();
          const data = JSON.parse(text);
          message = data.detail || message;
        } catch {
          // Keep the default error message
        }
      }

      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const downloadResult = () => {
    if (!result) return;

    const link = document.createElement("a");
    link.href = result;
    link.download = "floodvision-prediction.png";
    link.click();
  };

  const menu = [
    { id: "dashboard", name: "Dashboard", icon: LayoutDashboard },
    { id: "detection", name: "Flood Detection", icon: Waves },
    { id: "models", name: "ML Models", icon: BrainCircuit },
    { id: "analytics", name: "Analytics", icon: BarChart3 },
    { id: "history", name: "History", icon: History }
  ];

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="logo">
          <div className="logo-icon"><Waves size={25} /></div>
          <div>
            <h2>FloodVision</h2>
            <span>AI Intelligence Platform</span>
          </div>
        </div>

        <div className="menu-label">WORKSPACE</div>

        <nav className="navigation">
          {menu.map(item => {
            const Icon = item.icon;

            return (
              <button
                key={item.id}
                className={`nav-item ${page === item.id ? "active" : ""}`}
                onClick={() => setPage(item.id)}
              >
                <Icon size={19} />
                {item.name}
              </button>
            );
          })}
        </nav>

        <div className="sidebar-bottom">
          <span className="status-dot" />
          ML Backend Connected
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <span className="breadcrumb">FloodVision AI / {page}</span>
            <h1>
              {page === "dashboard" ? "Overview Dashboard" :
               page === "detection" ? "Flood Detection" :
               page === "models" ? "Machine Learning Models" :
               page === "analytics" ? "Model Analytics" :
               "Prediction History"}
            </h1>
          </div>
          <div className="top-badge">
            <span className="status-dot" />
            AI System
          </div>
        </header>

        <div className="content">

          {error && (
            <div className="error">
              <AlertCircle size={18} />
              {error}
            </div>
          )}

          {page === "dashboard" && (
            <>
              <section className="hero">
                <div>
                  <span className="eyebrow">AI-POWERED FLOOD SEGMENTATION</span>
                  <h2>Intelligent Flood Detection</h2>
                  <p>
                    Analyze flood imagery using your trained
                    machine learning models and visualize
                    predicted flooded regions.
                  </p>
                  <button
                    className="primary-btn"
                    onClick={() => setPage("detection")}
                  >
                    <Waves size={18} />
                    Start Detection
                  </button>
                </div>
                <BrainCircuit size={100} className="hero-icon" />
              </section>

              <div className="stats-grid">
                <StatCard
                  title="Available Models"
                  value={models.length}
                  icon={BrainCircuit}
                  subtitle="Loaded from FastAPI"
                />
                <StatCard
                  title="Predictions This Session"
                  value={predictionCount}
                  icon={Activity}
                  subtitle="Current browser session"
                />
                <StatCard
                  title="Last Flood Coverage"
                  value={floodPercentage === null ? "--" : `${floodPercentage}%`}
                  icon={Waves}
                  subtitle="Predicted image coverage"
                />
                <StatCard
                  title="Last Inference Time"
                  value={inferenceTime === null ? "--" : `${inferenceTime.toFixed(0)} ms`}
                  icon={Clock3}
                  subtitle="Latest prediction"
                />
              </div>

              <div className="panel">
                <h3>Project Overview</h3>
                <p>
                  FloodVision AI integrates supervised machine learning,
                  feature extraction, clustering and knowledge distillation
                  for image-based flood segmentation.
                </p>
                <button
                  className="secondary-btn"
                  onClick={() => setPage("models")}
                >
                  Explore Available Models
                </button>
              </div>
            </>
          )}

          {page === "detection" && (
            <div className="detection-layout">
              <div className="panel">
                <h3>Upload Flood Image</h3>
                <p>Select an image to analyze using your ML model.</p>

                <label className="upload-area">
                  <UploadCloud size={38} />
                  <strong>Click to upload image</strong>
                  <span>PNG, JPG or WebP · Max 10 MB</span>
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    onChange={handleFile}
                    hidden
                  />
                </label>

                {preview && (
                  <img
                    className="preview-image"
                    src={preview}
                    alt="Uploaded flood scene"
                  />
                )}

                <label className="field-label">Detection Model</label>
                <select
                  value={selectedModel}
                  onChange={e => setSelectedModel(e.target.value)}
                >
                  {models.length === 0 && (
                    <option value="RF-T">RF-T</option>
                  )}
                  {models.map(model => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>

                <button
                  className="primary-btn full-width"
                  onClick={predictFlood}
                  disabled={loading || !file || models.length === 0}
                >
                  <Play size={17} />
                  {loading ? "Analyzing Image..." : "Run Flood Detection"}
                </button>
              </div>

              <div className="panel result-panel">
                <div className="result-heading">
                  <h3>Segmentation Results</h3>
                  {result && <CheckCircle2 size={20} color="#22c55e" />}
                </div>

                {!result ? (
                  <div className="empty-result">
                    <Waves size={55} />
                    <h3>No Prediction Yet</h3>
                    <p>Upload an image and run flood detection.</p>
                  </div>
                ) : (
                  <>
                    <img
                      src={result}
                      alt="Original image and predicted flood overlay"
                      className="result-image"
                    />
                    <p className="image-caption">
                      Original image (left) · Predicted flood overlay (right)
                    </p>

                    <div className="result-stats">
                      <StatCard
                        title="Predicted Flood Coverage"
                        value={`${floodPercentage}%`}
                        icon={Waves}
                      />
                      <StatCard
                        title="Inference Time"
                        value={`${inferenceTime?.toFixed(1)} ms`}
                        icon={Clock3}
                      />
                    </div>

                    <button
                      className="secondary-btn"
                      onClick={downloadResult}
                    >
                      <Download size={17} />
                      Download Result
                    </button>
                  </>
                )}
              </div>
            </div>
          )}

          {page === "models" && (
            <div className="panel">
              <h3>Available Trained Models ({models.length})</h3>
              <p>Models loaded from your existing ML notebook.</p>
              <div className="model-grid">
                {models.map(model => (
                  <div key={model} className="model-card">
                    <BrainCircuit size={22} />
                    <strong>{model}</strong>
                    <button
                      onClick={() => {
                        setSelectedModel(model);
                        setPage("detection");
                      }}
                    >
                      Use Model
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {page === "analytics" && <Analytics />}

          {page === "history" && (
            <div className="panel">
              <h3>Prediction History</h3>
              <p>
                Prediction storage will be added in the database
                integration phase. Current session predictions:
                {" "}{predictionCount}
              </p>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}

function StatCard({ title, value, icon: Icon, subtitle }) {
  return (
    <div className="stat-card">
      <div className="stat-header">
        <span>{title}</span>
        <Icon size={20} />
      </div>
      <h2>{value}</h2>
      {subtitle && <small>{subtitle}</small>}
    </div>
  );
}
