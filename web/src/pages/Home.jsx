import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import Checker from "../components/Checker.jsx";
import CutScale from "../components/CutScale.jsx";
import { fetchThresholds } from "../api.js";

const METHODS = [
  {
    name: "DeepFakes",
    text: "Identity swap from the FaceForensics++ release. The study treats it as a fake class.",
  },
  {
    name: "Face2Face",
    text: "Reenactment that drives one face with another person’s expression.",
  },
  {
    name: "FaceSwap",
    text: "A second swap method in the same official split, scored as fake.",
  },
  {
    name: "NeuralTextures",
    text: "The neural-texture manipulation in the release. Also a fake class for this network.",
  },
];

const STEPS = [
  {
    title: "Decode",
    text: "An image is one frame. A short video is read in order, and clips longer than 15 seconds are refused.",
  },
  {
    title: "Crop",
    text: "The largest face is enlarged by 1.3. If the detector misses, the frame is skipped. A miss is never replaced with the full picture.",
  },
  {
    title: "Score",
    text: "Each crop is resized to 299 and passed through Xception. The clip score is the mean fake-class softmax of up to 16 crops.",
  },
  {
    title: "Cut",
    text: "Two frozen validation cuts place that score. There is no second calibration on top of the softmax.",
  },
];

export default function Home() {
  const [cuts, setCuts] = useState(null);
  const [cutError, setCutError] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetchThresholds()
      .then((payload) => {
        if (!cancelled) setCuts(payload);
      })
      .catch((err) => {
        if (!cancelled) setCutError(err instanceof Error ? err.message : "The cuts could not be loaded.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main>
      <section className="hero">
        <div className="frame hero-grid">
          <div>
            <p className="eyebrow">Local panel demo</p>
            <h1>Check a face still or a short clip.</h1>
            <p className="lede">
              Drop a file. VeriShield crops the largest face, scores it with this study’s XceptionNet, and places the
              mean fake-class softmax on two validation cuts: accept, manual review, or reject.
            </p>
            <div className="hero-actions">
              <a className="primary" href="#drop">
                Start a check
              </a>
              <Link className="ghost" to="/check">
                Open the checker
              </Link>
            </div>
            <p className="meta-line">Image or short video · mean fake-class softmax · no invented score</p>
          </div>
          <Checker heading="Select an image or a short video." />
        </div>
      </section>

      <section className="band" id="decisions">
        <div className="frame">
          <div className="section-head">
            <p className="eyebrow on-dark">Validation cuts</p>
            <h2>Three outcomes, two numbers.</h2>
            <p>
              The paper’s accuracy stays a 0.5 cut. This demo uses the separate pair frozen in{" "}
              <code>configs/thresholds-val.json</code>. Accept is below τ1. Manual review runs until τ2. Reject starts
              at τ2.
            </p>
          </div>
          {cutError ? <p className="band-error">{cutError}</p> : null}
          {cuts ? (
            <>
              <div className="decision-grid">
                <article>
                  <h3>Accept</h3>
                  <p className="cut">score &lt; {cuts.tau1}</p>
                  <p>{cuts.bands.accept}</p>
                </article>
                <article>
                  <h3>Manual review</h3>
                  <p className="cut">
                    {cuts.tau1} ≤ score &lt; {cuts.tau2}
                  </p>
                  <p>{cuts.bands.manual_review}</p>
                </article>
                <article>
                  <h3>Reject</h3>
                  <p className="cut">score ≥ {cuts.tau2}</p>
                  <p>{cuts.bands.reject}</p>
                </article>
              </div>
              <CutScale tau1={cuts.tau1} tau2={cuts.tau2} start={0.8} end={0.92} caption="Where the two cuts sit. This is not a file score." />
            </>
          ) : null}
        </div>
      </section>

      <section className="sheet" id="study">
        <div className="frame">
          <div className="section-head">
            <p className="eyebrow">What the study measures</p>
            <h2>Four manipulations, one face crop.</h2>
            <p>
              The network is trained on FaceForensics++ methods at raw quality and at H.264 QP 23 and QP 40, using the
              official 720/140/140 video split. The demo does not download that dataset when you open this page.
            </p>
          </div>
          <div className="method-grid">
            {METHODS.map((method) => (
              <article key={method.name}>
                <h3>{method.name}</h3>
                <p>{method.text}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="sheet alt" id="method">
        <div className="frame">
          <div className="section-head">
            <p className="eyebrow">How a file is read</p>
            <h2>The same crop and the same softmax as the study.</h2>
          </div>
          <ol className="steps">
            {STEPS.map((step, index) => (
              <li key={step.title}>
                <span>0{index + 1}</span>
                <div>
                  <h3>{step.title}</h3>
                  <p>{step.text}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="sheet" id="questions">
        <div className="frame faq-grid">
          <div>
            <p className="eyebrow">Notes for the panel</p>
            <h2>What this page will not do.</h2>
          </div>
          <div className="faqs">
            <details open>
              <summary>What does the score mean?</summary>
              <p>
                It is the mean probability of the fake class after a two-class softmax, averaged over the face crops
                that were found. It is not a calibrated likelihood from a second model.
              </p>
            </details>
            <details>
              <summary>Why might a real face still be rejected?</summary>
              <p>
                The cuts were chosen on a small validation pair so the demo can show three outcomes. A score at or
                above τ2 is a reject on that rule, even when a person would call the picture real.
              </p>
            </details>
            <details>
              <summary>What if no face is found?</summary>
              <p>The API returns an error and does not fill in a score. The full frame is not passed to the network.</p>
            </details>
            <details>
              <summary>What if the checkpoint is missing?</summary>
              <p>
                Set <code>CHECKPOINT_PATH</code> to the real weight file. If that path is empty or the file is absent,
                the upload returns an error and the page does not show a score.
              </p>
            </details>
            <details>
              <summary>How long can the video be?</summary>
              <p>
                Up to 15 seconds. The score still uses at most 16 face crops from the frames the decoder reads first.
              </p>
            </details>
          </div>
        </div>
      </section>
    </main>
  );
}
