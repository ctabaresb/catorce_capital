/**
 * Catorce Capital - API Proxy Worker
 *
 * Sits between the public dashboard and AWS API Gateway.
 * Injects the API key server-side so it never appears in the browser.
 *
 * SETUP (one time, ~5 minutes):
 *   1. Go to workers.cloudflare.com → Create Worker → paste this code → Deploy
 *   2. In the Worker settings → Variables → Add Secret:
 *        Name:  API_KEY
 *        Value: (your x-api-key from: cd infra/terraform && tofu output api_key)
 *   3. Copy your Worker URL (e.g. https://catorce-proxy.yourname.workers.dev)
 *   4. Paste it as WORKER_URL in dashboard_public.html
 */

const UPSTREAM = "https://j44cjs4ozj.execute-api.us-east-1.amazonaws.com/v1";
const ALLOWED_PATHS = ["/health", "/strategies", "/simulations", "/universe", "/backtest"];

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
          "Access-Control-Max-Age": "86400",
        },
      });
    }

    if (request.method !== "GET") {
      return new Response("Method not allowed", { status: 405 });
    }

    const url = new URL(request.url);
    const path = url.pathname + url.search;

    const allowed = ALLOWED_PATHS.some(p => url.pathname === p || url.pathname.startsWith(p + "/"));
    if (!allowed) {
      return new Response("Not found", { status: 404 });
    }

    const upstream = await fetch(UPSTREAM + path, {
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
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "public, max-age=300",
      },
    });
  },
};
