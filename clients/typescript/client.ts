/**
 * Typed client for the dipcatcher read-only research API.
 *
 * Zero runtime dependencies: paths and response types are derived from the
 * generated `schema.d.ts` (openapi-typescript) so this client cannot drift
 * from the committed `openapi.json`. Works in any environment with `fetch`
 * (Node 18+, browsers).
 *
 * The API is auth-less on loopback by default. When the server runs with
 * `RESEARCH_API_KEY` set, pass `apiKey` — it is sent as `X-API-Key`.
 */

import type { components, paths } from "./schema.d.ts";

export type HealthResponse = components["schemas"]["HealthResponse"];
export type RunSummary = components["schemas"]["RunSummary"];
export type RunListResponse = components["schemas"]["RunListResponse"];
export type RunDetail = components["schemas"]["RunDetail"];
export type RunMarkdownDetail = components["schemas"]["RunMarkdownDetail"];
export type ReceiptSummary = components["schemas"]["ReceiptSummary"];
export type ReceiptListResponse = components["schemas"]["ReceiptListResponse"];
export type ReceiptDetail = components["schemas"]["ReceiptDetail"];
export type ResultFile = components["schemas"]["ResultFile"];
export type ResultSummary = components["schemas"]["ResultSummary"];
export type ResultListResponse = components["schemas"]["ResultListResponse"];
export type ResultDetail = components["schemas"]["ResultDetail"];
export type VerifierVersionSummary =
  components["schemas"]["VerifierVersionSummary"];
export type VerifierVersionListResponse =
  components["schemas"]["VerifierVersionListResponse"];
export type VerifierVersionDetail =
  components["schemas"]["VerifierVersionDetail"];
export type VerifierRunSummary = components["schemas"]["VerifierRunSummary"];
export type VerifierRunListResponse =
  components["schemas"]["VerifierRunListResponse"];
export type VerifierRunDetail = components["schemas"]["VerifierRunDetail"];
export type ArtifactSummary = components["schemas"]["ArtifactSummary"];
export type ArtifactListResponse =
  components["schemas"]["ArtifactListResponse"];
export type ArtifactDetail = components["schemas"]["ArtifactDetail"];
export type VerificationResult = components["schemas"]["VerificationResult"];

/** JSON body of a 200 GET response for a path in the OpenAPI spec. */
type GetJson<P extends keyof paths> =
  paths[P]["get"] extends { responses: { 200: { content: { "application/json": infer B } } } }
    ? B
    : never;

export interface ResearchApiClientOptions {
  /** Default `http://127.0.0.1:8010`. */
  baseUrl?: string;
  /** Sent as `X-API-Key` when the server runs with RESEARCH_API_KEY set. */
  apiKey?: string;
  /** Injectable fetch implementation (defaults to global fetch). */
  fetch?: typeof fetch;
}

export class ResearchApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, detail: unknown) {
    const message =
      typeof detail === "object" && detail !== null && "detail" in detail
        ? String((detail as { detail: unknown }).detail)
        : `HTTP ${status}`;
    super(message);
    this.name = "ResearchApiError";
    this.status = status;
    this.detail = detail;
  }
}

export class ResearchApiClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ResearchApiClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "http://127.0.0.1:8010").replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.fetchImpl = options.fetch ?? fetch;
  }

  private async get<T>(path: string, query?: Record<string, unknown>): Promise<T> {
    const url = new URL(this.baseUrl + path);
    if (query) {
      for (const [key, value] of Object.entries(query)) {
        if (value !== undefined && value !== null) {
          url.searchParams.set(key, String(value));
        }
      }
    }
    const headers: Record<string, string> = { Accept: "application/json" };
    if (this.apiKey) {
      headers["X-API-Key"] = this.apiKey;
    }
    const resp = await this.fetchImpl(url.toString(), { headers });
    const body: unknown = await resp.json().catch(() => null);
    if (!resp.ok) {
      throw new ResearchApiError(resp.status, body);
    }
    return body as T;
  }

  health(): Promise<GetJson<"/health">> {
    return this.get("/health");
  }

  listRuns(params: { limit?: number; offset?: number } = {}): Promise<GetJson<"/runs">> {
    return this.get("/runs", { limit: params.limit, offset: params.offset });
  }

  latestRun(): Promise<GetJson<"/runs/latest">> {
    return this.get("/runs/latest");
  }

  getRun(runId: string): Promise<GetJson<"/runs/{run_id}">> {
    return this.get(`/runs/${encodeURIComponent(runId)}`);
  }

  getRunMarkdown(runId: string): Promise<GetJson<"/runs/{run_id}/markdown">> {
    return this.get(`/runs/${encodeURIComponent(runId)}/markdown`);
  }

  verifyRun(runId: string): Promise<GetJson<"/runs/{run_id}/verification">> {
    return this.get(`/runs/${encodeURIComponent(runId)}/verification`);
  }

  listReceipts(): Promise<GetJson<"/receipts">> {
    return this.get("/receipts");
  }

  getReceipt(receiptId: string): Promise<GetJson<"/receipts/{receipt_id}">> {
    return this.get(`/receipts/${encodeURIComponent(receiptId)}`);
  }

  getReceiptByHash(sha256: string): Promise<GetJson<"/receipts/by-hash/{digest}">> {
    return this.get(`/receipts/by-hash/${encodeURIComponent(sha256)}`);
  }

  verifyReceipt(receiptId: string): Promise<GetJson<"/receipts/{receipt_id}/verification">> {
    return this.get(`/receipts/${encodeURIComponent(receiptId)}/verification`);
  }

  listResults(): Promise<GetJson<"/results">> {
    return this.get("/results");
  }

  getResult(kind: string, name: string): Promise<GetJson<"/results/{kind}/{name}">> {
    return this.get(`/results/${encodeURIComponent(kind)}/${encodeURIComponent(name)}`);
  }

  verifyResult(
    kind: string,
    name: string,
  ): Promise<GetJson<"/results/{kind}/{name}/verification">> {
    return this.get(
      `/results/${encodeURIComponent(kind)}/${encodeURIComponent(name)}/verification`,
    );
  }

  listVerifierVersions(): Promise<GetJson<"/verifier/versions">> {
    return this.get("/verifier/versions");
  }

  getVerifierVersion(version: string): Promise<GetJson<"/verifier/versions/{version}">> {
    return this.get(`/verifier/versions/${encodeURIComponent(version)}`);
  }

  listVerifierRuns(): Promise<GetJson<"/verifier/runs">> {
    return this.get("/verifier/runs");
  }

  getVerifierRun(runLogId: string): Promise<GetJson<"/verifier/runs/{run_log_id}">> {
    return this.get(`/verifier/runs/${encodeURIComponent(runLogId)}`);
  }

  listArtifacts(): Promise<GetJson<"/artifacts">> {
    return this.get("/artifacts");
  }

  getArtifact(artifactId: string): Promise<GetJson<"/artifacts/{artifact_id}">> {
    // artifact ids may contain `/` (nested artifacts); keep slashes.
    const id = artifactId.split("/").map(encodeURIComponent).join("/");
    return this.get(`/artifacts/${id}`);
  }
}
