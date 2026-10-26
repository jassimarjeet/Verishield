function formatTau(value) {
  return value.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}

function clamp(value, start, end) {
  return Math.min(end, Math.max(start, value));
}

function percent(value, start, end) {
  if (end === start) return 0;
  return ((clamp(value, start, end) - start) / (end - start)) * 100;
}

export default function CutScale({ tau1, tau2, score = null, start = 0, end = 1, caption }) {
  const scorePct = typeof score === "number" ? percent(score, start, end) : null;
  const outside =
    typeof score === "number" && (score < start || score > end) ? (score < start ? "left" : "right") : null;

  return (
    <figure className="scale">
      {caption ? <figcaption>{caption}</figcaption> : null}
      <div className="scale-track" aria-hidden="true">
        <span className="scale-zone accept" style={{ left: `${percent(start, start, end)}%`, width: `${percent(tau1, start, end)}%` }} />
        <span
          className="scale-zone review"
          style={{
            left: `${percent(tau1, start, end)}%`,
            width: `${Math.max(0, percent(tau2, start, end) - percent(tau1, start, end))}%`,
          }}
        />
        <span
          className="scale-zone reject"
          style={{
            left: `${percent(tau2, start, end)}%`,
            width: `${Math.max(0, 100 - percent(tau2, start, end))}%`,
          }}
        />
        <i className="scale-cut" style={{ left: `${percent(tau1, start, end)}%` }} />
        <i className="scale-cut" style={{ left: `${percent(tau2, start, end)}%` }} />
        {scorePct !== null ? <b className={`scale-score ${outside ? "outside" : ""}`} style={{ left: `${scorePct}%` }} /> : null}
      </div>
      <div className="scale-labels">
        <span>{start.toFixed(2)}</span>
        <span>τ1 {formatTau(tau1)}</span>
        <span>τ2 {formatTau(tau2)}</span>
        <span>{end.toFixed(2)}</span>
      </div>
      {outside ? (
        <p className="scale-note">The score sits {outside === "left" ? "below" : "above"} this window.</p>
      ) : null}
    </figure>
  );
}
