import { Link } from "react-router-dom";
import Checker from "../components/Checker.jsx";

export default function Check() {
  return (
    <main className="check-page">
      <div className="frame check-grid">
        <div>
          <p className="eyebrow">Upload</p>
          <h1>Read one file against the frozen cuts.</h1>
          <p className="lede">
            Choose an image or a short video. After the API answers, the card shows the score, the decision, and a
            short account of accept, manual review, and reject.
          </p>
          <ol className="plain-steps">
            <li>Accept is strictly below τ1.</li>
            <li>Manual review is from τ1 until τ2.</li>
            <li>Reject is τ2 and anything above it.</li>
          </ol>
          <p className="meta-line">
            The cuts themselves are on the <Link to="/#decisions">landing page</Link>.
          </p>
        </div>
        <Checker heading="Drop the file, then read the band." />
      </div>
    </main>
  );
}
