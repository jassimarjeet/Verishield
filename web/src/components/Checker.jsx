import { useEffect, useId, useMemo, useState } from "react";
import { detectFile, fetchHealth } from "../api.js";
import CutScale from "./CutScale.jsx";

const KINDS = [
  {
    id: "image",
    label: "Image",
    accept: "image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp",
    hint: "JPEG, PNG, or WebP · up to 12 MB",
    prompt: "Drop a face still here",
  },
  {
    id: "video",
    label: "Video",
    accept: "video/mp4,video/webm,video/quicktime,.mp4,.mov,.webm",
    hint: "MP4, MOV, or WebM · up to 15 seconds",
    prompt: "Drop a short face clip here",
  },
];

const DECISION_LABEL = {
  accept: "Accept",
  manual_review: "Manual review",
  reject: "Reject",
};

function kindForFile(file) {
  const name = file.name.toLowerCase();
  if (file.type.startsWith("video/") || /\.(mp4|mov|webm|mkv|avi)$/.test(name)) return "video";
  if (file.type.startsWith("image/") || /\.(jpe?g|png|webp)$/.test(name)) return "image";
  return null;
}

export default function Checker({ heading = "Submit a face still or a short clip." }) {
  const inputId = useId();
  const [kind, setKind] = useState("image");
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [apiState, setApiState] = useState("checking");
  const [dragOver, setDragOver] = useState(false);
  const active = KINDS.find((item) => item.id === kind) ?? KINDS[0];

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  useEffect(() => {
    let cancelled = false;
    fetchHealth()
      .then(() => {
        if (!cancelled) setApiState("ready");
      })
      .catch(() => {
        if (!cancelled) setApiState("down");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function resetReading() {
    setResult(null);
    setError("");
    setStatus("idle");
  }

  function takeFile(next) {
    if (!next) return;
    const detected = kindForFile(next);
    if (!detected) {
      setFile(null);
      setResult(null);
      setStatus("idle");
      setError("Use an image or a short video. This demo does not score other files.");
      return;
    }
    setKind(detected);
    setFile(next);
    setError("");
    setResult(null);
    setStatus("idle");
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (!file || status === "reading") return;
    setStatus("reading");
    setError("");
    setResult(null);
    try {
      const payload = await detectFile(file);
      setResult(payload);
      setStatus("ready");
    } catch (err) {
      setStatus("idle");
      setError(err instanceof Error ? err.message : "Detection did not return a score.");
    }
  }

  const decision = result?.decision;
  const bands = result?.bands ?? {};

  return (
    <div className="checker" id="drop" aria-labelledby="checker-title">
      <div className="checker-head">
        <p className="eyebrow">Face score</p>
        <h2 id="checker-title">{heading}</h2>
        <p className="lede tight">
          The API crops the largest face, runs the study checkpoint, and returns the mean fake-class softmax with
          the three-way cut.
        </p>
        <p className={`api-state ${apiState}`}>
          {apiState === "ready" && "API reachable on this machine."}
          {apiState === "down" && "API not reachable. Start it on port 8000 before submitting."}
          {apiState === "checking" && "Checking the API."}
        </p>
      </div>

      {result ? (
        <div className={`reading reading-${decision}`} role="status" aria-live="polite">
          <p className="eyebrow">Reading</p>
          <h3>{DECISION_LABEL[decision] ?? decision}</h3>
          <p className="score-figure">
            <span>Mean fake-class softmax</span>
            <strong>{Number(result.score).toFixed(6)}</strong>
          </p>
          <CutScale
            tau1={result.tau1}
            tau2={result.tau2}
            score={result.score}
            caption="Full range"
          />
          <CutScale
            tau1={result.tau1}
            tau2={result.tau2}
            score={result.score}
            start={0.8}
            end={0.92}
            caption="Window around the cuts"
          />
          <p className="explanation">{result.explanation}</p>
          <ul className="band-list">
            {["accept", "manual_review", "reject"].map((key) => (
              <li key={key} className={key === decision ? "is-active" : ""}>
                <strong>{DECISION_LABEL[key]}</strong>
                <span>{bands[key]}</span>
              </li>
            ))}
          </ul>
          <p className="crop-count">
            {result.n_frames
              ? `Scored ${result.n_frames} face ${result.n_frames === 1 ? "crop" : "crops"} from this ${result.media_kind}.`
              : null}
          </p>
          <button type="button" className="text-button" onClick={resetReading}>
            Check another file
          </button>
        </div>
      ) : (
        <form className="drop-form" onSubmit={onSubmit}>
          <div className="kind-tabs" role="tablist" aria-label="File kind">
            {KINDS.map((item) => (
              <button
                key={item.id}
                type="button"
                role="tab"
                aria-selected={kind === item.id}
                className={kind === item.id ? "is-selected" : ""}
                onClick={() => {
                  setKind(item.id);
                  setFile(null);
                  setError("");
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
          <label
            htmlFor={inputId}
            className={`dropzone ${dragOver ? "is-over" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragOver(false);
              takeFile(event.dataTransfer.files?.[0]);
            }}
          >
            <input
              id={inputId}
              type="file"
              accept={active.accept}
              onChange={(event) => takeFile(event.target.files?.[0])}
            />
            {previewUrl && file ? (
              kind === "video" ? (
                <video className="preview" src={previewUrl} muted controls />
              ) : (
                <img className="preview" src={previewUrl} alt="" />
              )
            ) : (
              <span className="drop-copy">
                <strong>{active.prompt}</strong>
                <span>{active.hint}</span>
              </span>
            )}
          </label>
          {file ? <p className="file-name">{file.name}</p> : null}
          {error ? (
            <p className="form-error" role="alert">
              {error}
            </p>
          ) : null}
          <div className="form-actions">
            <button className="primary" type="submit" disabled={!file || status === "reading"}>
              {status === "reading" ? "Reading face crops…" : "Read the score"}
            </button>
            {status === "reading" ? <p className="pending">No score yet. Waiting for the checkpoint.</p> : null}
          </div>
        </form>
      )}
    </div>
  );
}
