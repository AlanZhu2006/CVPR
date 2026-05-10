var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o, _p, _q, _r, _s, _t, _u, _v, _w, _x, _y, _z, _0;
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { extname, resolve, sep } from "node:path";
var nodeEnv = (_b = (_a = globalThis.process) === null || _a === void 0 ? void 0 : _a.env) !== null && _b !== void 0 ? _b : {};
var playbackControlTarget = (_c = nodeEnv.VITE_PLAYBACK_CONTROL_TARGET) !== null && _c !== void 0 ? _c : "http://127.0.0.1:".concat((_e = (_d = nodeEnv.WEB_PLAYBACK_CONTROL_PORT) !== null && _d !== void 0 ? _d : nodeEnv.CONTROL_PORT) !== null && _e !== void 0 ? _e : "8765");
var rosbridgeTarget = (_f = nodeEnv.VITE_ROSBRIDGE_TARGET) !== null && _f !== void 0 ? _f : "ws://127.0.0.1:".concat((_h = (_g = nodeEnv.WEB_ROSBRIDGE_PORT) !== null && _g !== void 0 ? _g : nodeEnv.ROSBRIDGE_PORT) !== null && _h !== void 0 ? _h : "9090");
var hifiBridgeTarget = (_j = nodeEnv.VITE_HIFI_BRIDGE_TARGET) !== null && _j !== void 0 ? _j : "http://127.0.0.1:".concat((_l = (_k = nodeEnv.WEB_HIFI_BRIDGE_PORT) !== null && _k !== void 0 ? _k : nodeEnv.HIFI_PORT) !== null && _l !== void 0 ? _l : "8876");
var hifiWindowTarget = (_m = nodeEnv.VITE_HIFI_WINDOW_TARGET) !== null && _m !== void 0 ? _m : "http://127.0.0.1:".concat((_p = (_o = nodeEnv.WEB_HIFI_WINDOW_PORT) !== null && _o !== void 0 ? _o : nodeEnv.HIFI_WINDOW_PORT) !== null && _p !== void 0 ? _p : "8877");
var isaacGaussianOnlineTarget = (_q = nodeEnv.VITE_ISAAC_GAUSSIAN_ONLINE_TARGET) !== null && _q !== void 0 ? _q : "http://127.0.0.1:".concat((_s = (_r = nodeEnv.WEB_ISAAC_GAUSSIAN_ONLINE_PORT) !== null && _r !== void 0 ? _r : nodeEnv.ISAAC_GAUSSIAN_ONLINE_PORT) !== null && _s !== void 0 ? _s : "8890");
var isaacGaussianMapperTarget = (_t = nodeEnv.VITE_ISAAC_GAUSSIAN_MAPPER_TARGET) !== null && _t !== void 0 ? _t : "http://127.0.0.1:".concat((_v = (_u = nodeEnv.WEB_ISAAC_GAUSSIAN_MAPPER_PORT) !== null && _u !== void 0 ? _u : nodeEnv.ISAAC_GAUSSIAN_MAPPER_PORT) !== null && _v !== void 0 ? _v : "8891");
var worldNavTarget = (_w = nodeEnv.VITE_WORLD_NAV_TARGET) !== null && _w !== void 0 ? _w : "http://127.0.0.1:".concat((_y = (_x = nodeEnv.WEB_WORLD_NAV_PORT) !== null && _x !== void 0 ? _x : nodeEnv.WORLD_NAV_PORT) !== null && _y !== void 0 ? _y : "8892");
var lingbotLiveAssetRoot = (_z = nodeEnv.VITE_LINGBOT_LIVE_ASSET_ROOT) !== null && _z !== void 0 ? _z : "";
var lingbotReal2simAssetRoot = (_0 = nodeEnv.VITE_LINGBOT_REAL2SIM_ASSET_ROOT) !== null && _0 !== void 0 ? _0 : "";
export default defineConfig({
    plugins: [
        react(),
        {
            name: "lingbot-live-assets",
            configureServer: function (server) {
                server.middlewares.use(function (request, response, next) {
                    var _a, _b;
                    var requestUrl = (_a = request.url) !== null && _a !== void 0 ? _a : "";
                    var livePrefix = "/lingbot-live-assets/live/";
                    var real2simPrefix = "/lingbot-live-assets/real2sim/";
                    var match = requestUrl.startsWith(livePrefix)
                        ? { root: lingbotLiveAssetRoot, prefix: livePrefix }
                        : requestUrl.startsWith(real2simPrefix)
                            ? { root: lingbotReal2simAssetRoot, prefix: real2simPrefix }
                            : null;
                    if (!(match === null || match === void 0 ? void 0 : match.root)) {
                        next();
                        return;
                    }
                    var relativePath = decodeURIComponent((_b = requestUrl.slice(match.prefix.length).split("?")[0]) !== null && _b !== void 0 ? _b : "");
                    var root = resolve(match.root);
                    if (match.prefix === livePrefix && relativePath === "rgb.mjpg") {
                        serveMjpeg(response, request, resolve(root, "rgb_preview/latest.jpg"));
                        return;
                    }
                    if (match.prefix === livePrefix && relativePath === "gaussian.mjpg") {
                        serveMjpeg(response, request, resolve(root, "gaussian_preview/latest.jpg"));
                        return;
                    }
                    var target = resolve(root, relativePath);
                    if (target !== root && !target.startsWith(root + sep)) {
                        response.statusCode = 403;
                        response.end("Forbidden");
                        return;
                    }
                    if (!existsSync(target) || !statSync(target).isFile()) {
                        response.statusCode = 404;
                        response.end("Not found");
                        return;
                    }
                    response.setHeader("Content-Type", contentTypeForPath(target));
                    response.setHeader("Cache-Control", cacheControlForAsset(match.prefix, target));
                    createReadStream(target).pipe(response);
                });
            }
        }
    ],
    server: {
        host: "0.0.0.0",
        port: 5173,
        proxy: {
            "/__playback_control": {
                target: playbackControlTarget,
                changeOrigin: true,
                rewrite: function (path) { return path.replace(/^\/__playback_control/, ""); }
            },
            "/__hifi_bridge": {
                target: hifiBridgeTarget,
                changeOrigin: true,
                rewrite: function (path) { return path.replace(/^\/__hifi_bridge/, ""); }
            },
            "/__hifi_window": {
                target: hifiWindowTarget,
                changeOrigin: true,
                rewrite: function (path) { return path.replace(/^\/__hifi_window/, ""); }
            },
            "/__rosbridge": {
                target: rosbridgeTarget,
                ws: true,
                changeOrigin: true,
                rewrite: function (path) { return path.replace(/^\/__rosbridge/, ""); }
            },
            "/__isaac_gaussian_online": {
                target: isaacGaussianOnlineTarget,
                changeOrigin: true,
                rewrite: function (path) { return path.replace(/^\/__isaac_gaussian_online/, ""); }
            },
            "/__isaac_gaussian_mapper": {
                target: isaacGaussianMapperTarget,
                changeOrigin: true,
                rewrite: function (path) { return path.replace(/^\/__isaac_gaussian_mapper/, ""); }
            },
            "/__world_nav": {
                target: worldNavTarget,
                changeOrigin: true,
                rewrite: function (path) { return path.replace(/^\/__world_nav/, ""); }
            }
        }
    },
    preview: {
        host: "0.0.0.0",
        port: 4173
    }
});
function serveMjpeg(response, request, latestJpegPath) {
    response.writeHead(200, {
        "Content-Type": "multipart/x-mixed-replace; boundary=lingbotframe",
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        Pragma: "no-cache",
        Connection: "close"
    });
    var closed = false;
    var sendFrame = function () {
        if (closed) {
            return;
        }
        if (!existsSync(latestJpegPath)) {
            return;
        }
        try {
            var frame = readFileSync(latestJpegPath);
            response.write("--lingbotframe\r\nContent-Type: image/jpeg\r\nContent-Length: ".concat(frame.length, "\r\n\r\n"));
            response.write(frame);
            response.write("\r\n");
        }
        catch (_a) {
            // The writer may be atomically replacing the file; the next tick will retry.
        }
    };
    var interval = setInterval(sendFrame, 120);
    request.on("close", function () {
        closed = true;
        clearInterval(interval);
    });
    sendFrame();
}
function contentTypeForPath(path) {
    switch (extname(path).toLowerCase()) {
        case ".json":
            return "application/json";
        case ".ply":
            return "application/octet-stream";
        case ".png":
            return "image/png";
        case ".jpg":
        case ".jpeg":
            return "image/jpeg";
        default:
            return "application/octet-stream";
    }
}
function cacheControlForAsset(prefix, path) {
    if (prefix.includes("/real2sim/") && extname(path).toLowerCase() !== ".json") {
        return "public, max-age=31536000, immutable";
    }
    return "no-store";
}
