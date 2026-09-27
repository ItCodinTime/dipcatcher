/** Hash-based routing — the app is static and must run from any mount point
 * (including subpaths where server rewrites are unavailable).
 */

import { useEffect, useState } from "react";

export type Route =
  | { name: "overview" }
  | { name: "strategy"; id: string }
  | { name: "receipts" }
  | { name: "receipt"; id: string };

export function parseHash(hash: string): Route {
  const cleaned = hash.replace(/^#/, "");
  const parts = cleaned.split("/").filter((p) => p.length > 0);
  if (parts.length === 0) return { name: "overview" };
  if (parts[0] === "strategy" && parts[1]) {
    return { name: "strategy", id: decodeURIComponent(parts[1]) };
  }
  if (parts[0] === "receipts" && parts.length === 1) {
    return { name: "receipts" };
  }
  if (parts[0] === "receipt" && parts[1]) {
    return { name: "receipt", id: decodeURIComponent(parts[1]) };
  }
  return { name: "overview" };
}

export function routeHref(route: Route): string {
  switch (route.name) {
    case "overview":
      return "#/";
    case "strategy":
      return `#/strategy/${encodeURIComponent(route.id)}`;
    case "receipts":
      return "#/receipts";
    case "receipt":
      return `#/receipt/${encodeURIComponent(route.id)}`;
  }
}

export function useHashRoute(): Route {
  const [route, setRoute] = useState<Route>(() =>
    parseHash(window.location.hash),
  );
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}
