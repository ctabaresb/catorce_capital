/**
 * Catorce Capital - API Proxy Worker
 *
 * Sits between the Access-gated dashboard and AWS API Gateway.
 * Mounted at https://catorcelabs.com/api/* via a Worker Route, so
 * Cloudflare Access (which gates catorcelabs.com) gates this Worker
 * too. There is no public *.workers.dev URL.
 *
 * Defense in depth: requests with a non-canonical Origin header are
 * rejected at the Worker layer in case Access is ever misconfigured.
 *
 * SETUP (one time):
 *   1. workers.cloudflare.com -> Create Worker -> paste this code -> Deploy
 *   2. Worker -> Settings -> Variables and Secrets -> Add Secret
 *        Name:  API_KEY
 *        Value: cd infra/terraform && tofu output -raw api_key
 *   3. Worker -> Settings -> Domains & Routes
 *        Add Route: Zone catorcelabs.com, Pattern catorcelabs.com/api/*
 *        Disable the workers.dev preview URL
 *   4. Cloudflare Zero Trust -> Access -> Applications
 *        Confirm the catorcelabs.com app's path scope covers /api/*
 *
 * REDEPLOY (after editing this file):
 *   Cloudflare Worker editor -> paste this file -> Save and Deploy
 */

const UPSTREAM = "https://j44cjs4ozj.execute-api.us-east-1.amazonaws.com/v1";
const ALLOWED_PATHS = ["/health", "/strategies", "/simulations", "/universe", "/backtest"];
const BASE_PATH = "/api";
const ALLOWED_ORIGIN = "https://catorcelabs.com";

export default {
  async fetch(request, env) {
    if (request.method !== "GET") {
      return new Response("Method not allowed", { status: 405 });
    }

    // Same-origin browser fetches send Origin only on cross-origin or
    // non-GET requests, but if it IS sent we require the canonical value.
    // curl/scripted requests with a wrong Origin get rejected here even
    // if they somehow get past Access.
    const origin = request.headers.get("Origin");
    if (origin && origin !== ALLOWED_ORIGIN) {
      return new Response("Forbidden", { status: 403 });
    }

    // Strip the route mount prefix so the upstream path matches the
    // API Gateway routes. catorcelabs.com/api/health -> /health.
    const url = new URL(request.url);
    let pathname = url.pathname;
    if (pathname.startsWith(BASE_PATH + "/")) {
      pathname = pathname.slice(BASE_PATH.length);
    } else if (pathname === BASE_PATH) {
      pathname = "/";
    }

    const allowed = ALLOWED_PATHS.some(
      p => pathname === p || pathname.startsWith(p + "/")
    );
    if (!allowed) {
      return new Response("Not found", { status: 404 });
    }

    const upstream = await fetch(UPSTREAM + pathname + url.search, {
      headers: {
        "x-api-key": env.API_KEY,
        "Content-Type": "application/json",
      },
    });

    const body = await upstream.text();

    return new Response(body, {
      status: upstream.status,
      headers: {
        "Content-Type": "application/json",
        // Same-origin in normal use; tightening the legacy wildcard.
        "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
        "Cache-Control": "public, max-age=300",
      },
    });
  },
};
