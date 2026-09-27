import type { ReactNode } from "react";
import { fetchJson, fixturePath, validateIndex } from "./lib/fixtures";
import { useHashRoute } from "./lib/router";
import { useAsync } from "./lib/useJson";
import { Header } from "./components/Header";
import { OverviewPage } from "./components/OverviewPage";
import { StrategyPage } from "./components/StrategyPage";
import { ReceiptsPage } from "./components/ReceiptsPage";
import { ReceiptPage } from "./components/ReceiptPage";

export function App() {
  const route = useHashRoute();
  const index = useAsync("index", () =>
    fetchJson(fixturePath("index.json")).then(validateIndex),
  );

  if (index.loading) {
    return (
      <div className="app">
        <p className="loading">Loading fixture index…</p>
      </div>
    );
  }
  if (index.error || !index.data) {
    return (
      <div className="app">
        <p className="error">
          Could not load fixtures/index.json — {index.error ?? "empty"}.
          Regenerate with{" "}
          <code>uv run --no-sync python web/scripts/export_fixtures.py</code>.
        </p>
      </div>
    );
  }
  const idx = index.data;

  let page: ReactNode;
  switch (route.name) {
    case "overview":
      page = <OverviewPage index={idx} />;
      break;
    case "strategy":
      page = (
        <StrategyPage
          entry={idx.strategies.find((s) => s.id === route.id)}
          index={idx}
        />
      );
      break;
    case "receipts":
      page = <ReceiptsPage index={idx} />;
      break;
    case "receipt":
      page = (
        <ReceiptPage
          entry={idx.receipts.find((r) => r.id === route.id)}
          index={idx}
        />
      );
      break;
  }

  return (
    <div className="app">
      <Header route={route} />
      {page}
    </div>
  );
}
