import { routeHref, type Route } from "../lib/router";

export function Header({ route }: { route: Route }) {
  return (
    <header className="site-header">
      <div className="header-row">
        <a href={routeHref({ name: "overview" })} className="brand">
          dipcatcher <span className="brand-sub">research explorer</span>
        </a>
        <nav>
          <a
            href={routeHref({ name: "overview" })}
            aria-current={route.name === "overview" || route.name === "strategy" ? "page" : undefined}
          >
            Strategies
          </a>
          <a
            href={routeHref({ name: "receipts" })}
            aria-current={
              route.name === "receipts" || route.name === "receipt"
                ? "page"
                : undefined
            }
          >
            Receipts
          </a>
        </nav>
      </div>
      <div className="honesty-strip" data-testid="honesty-strip">
        RESEARCH ONLY — read-only view of sealed research receipts and
        committed backtest artifacts. Simulated evidence, not live trading
        performance. Authoritative verification:{" "}
        <code>dipcatcher verify-research</code>.
      </div>
    </header>
  );
}
