import { Link, NavLink } from "react-router-dom";

function Mark() {
  return (
    <svg className="mark" viewBox="0 0 32 32" aria-hidden="true">
      <path
        d="M16 4.2 26.5 8.6V16c0 6.1-4.2 10-10.5 11.8C9.7 26 5.5 22.1 5.5 16V8.6L16 4.2z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
      />
      <path d="M16 11v9.2M11.6 15.6h8.8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

export function SiteHeader() {
  return (
    <header className="site-header">
      <div className="frame header-row">
        <Link className="brand" to="/" aria-label="VeriShield home">
          <Mark />
          <span>
            <strong>VeriShield</strong>
            <small>Face-crop study</small>
          </span>
        </Link>
        <nav className="nav" aria-label="Primary">
          <NavLink to="/check">Check</NavLink>
          <a href="/#decisions">Decisions</a>
          <a href="/#study">Study</a>
          <a href="/#questions">Notes</a>
        </nav>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="frame footer-row">
        <div>
          <strong>VeriShield</strong>
          <p>A local panel demo for one face-crop Xception study. It does not screen audio.</p>
        </div>
        <div className="footer-links">
          <Link to="/check">Open the checker</Link>
          <a href="/#decisions">Read the cuts</a>
        </div>
      </div>
    </footer>
  );
}
