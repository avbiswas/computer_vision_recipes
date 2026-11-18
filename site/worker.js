// cv.neuralavb.com
// Static pages come from dist/ (built in CI from git) and never reach this code.
// /r2/<key> serves the big, git-ignored files (models, data, images, videos) from the R2 bucket.
//  * The edge cache key includes the R2 etag, so a re-uploaded file is served at once (no stale cache).
//  * Browsers revalidate with If-None-Match and get a 304 when nothing changed (no re-download of big models).
//  * HTTP Range requests are supported (video seeking needs them).
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (!url.pathname.startsWith("/r2/")) return env.ASSETS.fetch(request);
    if (request.method !== "GET" && request.method !== "HEAD") return new Response("method not allowed", { status: 405 });

    const key = decodeURIComponent(url.pathname.slice(4));
    const head = await env.BUCKET.head(key);
    if (!head) return new Response("not found", { status: 404 });

    const headers = new Headers();
    head.writeHttpMetadata(headers);
    headers.set("etag", head.httpEtag);
    headers.set("accept-ranges", "bytes");
    headers.set("cache-control", "public, no-cache");  // browser keeps it, but asks each time (cheap 304)
    if (!headers.has("content-type")) headers.set("content-type", "application/octet-stream");

    const inm = request.headers.get("if-none-match");
    if (inm && inm.split(",").map((s) => s.trim().replace(/^W\//, "")).includes(head.httpEtag)) {
      return new Response(null, { status: 304, headers });
    }
    if (request.method === "HEAD") {
      headers.set("content-length", String(head.size));
      return new Response(null, { headers });
    }

    const cache = caches.default;
    const cacheKey = new Request(`${url.origin}${url.pathname}?etag=${encodeURIComponent(head.etag)}`);
    const range = request.headers.get("range");

    if (range) {
      // the edge cache answers Range requests from a cached full copy; otherwise read just the range from R2
      const hit = await cache.match(new Request(cacheKey, { headers: { range } }));
      if (hit) return hit;
      const obj = await env.BUCKET.get(key, { range: request.headers });
      if (!obj) return new Response("not found", { status: 404 });
      if (obj.range) {
        const r = obj.range;
        const start = r.offset ?? (r.suffix != null ? obj.size - r.suffix : 0);
        const len = r.length ?? (r.suffix ?? obj.size - start);
        headers.set("content-range", `bytes ${start}-${start + len - 1}/${obj.size}`);
        headers.set("content-length", String(len));
        return new Response(obj.body, { status: 206, headers });
      }
      headers.set("content-length", String(obj.size));
      return new Response(obj.body, { headers });
    }

    const hit = await cache.match(cacheKey);
    if (hit) return new Response(hit.body, { headers });
    const obj = await env.BUCKET.get(key);
    if (!obj) return new Response("not found", { status: 404 });
    headers.set("content-length", String(obj.size));
    const [toClient, toCache] = obj.body.tee();  // stream to the client and into the edge cache together
    const cacheHeaders = new Headers(headers);
    cacheHeaders.set("cache-control", "public, max-age=31536000, immutable");  // safe: the key changes with the etag
    ctx.waitUntil(cache.put(cacheKey, new Response(toCache, { headers: cacheHeaders })));
    return new Response(toClient, { headers });
  },
};
