/**
 * The HTTP client.
 *
 * Thin, but it owns two things worth doing once rather than at every call
 * site: turning a FastAPI error body into a message a person can read, and
 * polling long-running jobs.
 *
 * FastAPI puts the useful text in `detail`, which is a string for a raised
 * `HTTPException` and a list of field objects for a validation failure.
 * Rendering the latter with `JSON.stringify` — the obvious thing — produces
 * a wall of schema noise in front of the one sentence that matters.
 */

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

function readDetail(body, status) {
  if (!body) return `Request failed (${status})`;
  const d = body.detail ?? body.message ?? body.error;
  if (typeof d === 'string') return d;
  if (Array.isArray(d) && d.length) {
    const first = d[0];
    const field = Array.isArray(first?.loc) ? first.loc.slice(-1)[0] : null;
    const msg = first?.msg || 'is invalid';
    return field ? `${field}: ${msg}` : msg;
  }
  return `Request failed (${status})`;
}

export async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(path, opts);
  } catch (err) {
    // A failed fetch is almost always the server having stopped, and
    // "Failed to fetch" tells nobody that.
    throw new ApiError('Cannot reach the engine. Is `mixengine serve` running?',
                       0, null);
  }

  const type = res.headers.get('content-type') || '';
  const body = type.includes('application/json') ? await res.json().catch(() => null)
                                                 : null;
  if (!res.ok) throw new ApiError(readDetail(body, res.status), res.status, body);
  return body;
}

export function postForm(path, fields) {
  const fd = new FormData();
  for (const [k, v] of Object.entries(fields)) {
    if (v !== null && v !== undefined && v !== '') fd.append(k, v);
  }
  return api(path, { method: 'POST', body: fd });
}

/**
 * Poll a job until it settles.
 *
 * Backs off from 400 ms to 2 s. A render takes tens of seconds to minutes,
 * and polling it four times a second for three minutes is 700 requests to
 * learn what a dozen would have. Backing off keeps the early stages — where
 * the stage name changes quickly and the user is watching — responsive.
 */
export function pollJob(id, { onProgress, signal } = {}) {
  return new Promise((resolve, reject) => {
    let delay = 400;
    let cancelled = false;
    signal?.addEventListener('abort', () => { cancelled = true; reject(new Error('cancelled')); });

    const tick = async () => {
      if (cancelled) return;
      let job;
      try {
        job = await api(`/api/jobs/${encodeURIComponent(id)}`);
      } catch (err) {
        // A single failed poll is not a failed job; the server may be busy.
        setTimeout(tick, Math.min(delay * 2, 4000));
        return;
      }
      onProgress?.(job);
      if (job.status === 'done') return resolve(job.result ?? {});
      if (job.status === 'failed') return reject(new ApiError(job.error || 'The job failed', 0, job));
      delay = Math.min(delay * 1.25, 2000);
      setTimeout(tick, delay);
    };
    tick();
  });
}
