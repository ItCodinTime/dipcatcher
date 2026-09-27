import { useEffect, useState } from "react";

export interface AsyncData<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** Tiny fetch hook: loads `loader` whenever `key` changes. */
export function useAsync<T>(
  key: string | null,
  loader: (() => Promise<T>) | null,
): AsyncData<T> {
  const [state, setState] = useState<AsyncData<T>>({
    data: null,
    error: null,
    loading: loader !== null,
  });

  useEffect(() => {
    if (loader === null || key === null) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    let cancelled = false;
    setState({ data: null, error: null, loading: true });
    loader()
      .then((data) => {
        if (!cancelled) setState({ data, error: null, loading: false });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setState({
            data: null,
            error: err instanceof Error ? err.message : String(err),
            loading: false,
          });
        }
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return state;
}
