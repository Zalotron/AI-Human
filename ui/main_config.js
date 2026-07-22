// Proceso principal de Electron para el EDITOR DE LIMITES (Config Ragdoll).
// Lanza config_ragdoll_server.py (puerto 8771) y abre una ventana apuntando al mismo.
// Es una app APARTE de la viz principal (main.js / server.py, puerto 8770).
const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const PYTHON = path.join(ROOT, "venv", "Scripts", "python.exe");
const URL = "http://127.0.0.1:8771";

let py = null;
let win = null;

function startServer() {
  py = spawn(PYTHON, ["config_ragdoll_server.py"], { cwd: ROOT, stdio: "inherit" });
  py.on("error", (e) => console.error("[config] no se pudo iniciar Python:", e));
}

function waitForServer(cb, tries = 0) {
  http.get(URL, (res) => { res.destroy(); cb(); })
    .on("error", () => {
      if (tries > 150) return cb();
      setTimeout(() => waitForServer(cb, tries + 1), 150);
    });
}

function createWindow() {
  win = new BrowserWindow({
    width: 1440, height: 900, backgroundColor: "#0d0d0f",
    title: "AI Human — Config Ragdoll (límites)", autoHideMenuBar: true,
  });
  win.maximize();              // arranca MAXIMIZADA (con barra de titulo, no fullscreen sin bordes)
  win.loadURL(URL);
}

function killPy() { if (py) { try { py.kill(); } catch (e) {} py = null; } }

app.whenReady().then(() => { startServer(); waitForServer(createWindow); });
app.on("window-all-closed", () => { killPy(); app.quit(); });
app.on("before-quit", killPy);
