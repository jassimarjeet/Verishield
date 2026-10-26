import { Route, Routes } from "react-router-dom";
import Check from "./pages/Check.jsx";
import Home from "./pages/Home.jsx";
import { SiteFooter, SiteHeader } from "./components/SiteChrome.jsx";

export default function App() {
  return (
    <>
      <SiteHeader />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/check" element={<Check />} />
      </Routes>
      <SiteFooter />
    </>
  );
}
