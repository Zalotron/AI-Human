// Proceso principal de Electron.
// Lanza el servidor Python (server.py) y abre una ventana apuntando al mismo.
// El servidor corre la fisica MuJoCo + la politica y streamea el estado 3D.
const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const path = require("path");

const ROOT = path.join(__dirname, "..");                       // carpeta del proyecto
const PYTHON = path.join(ROOT, "venv", "Scripts", "python.exe");
const URL = "http://127.0.0.1:8770";

let py = null;
let win = null;

function startServer() {
  py = spawn(PYTHON, ["server.py"], { cwd: ROOT, stdio: "inherit" });
  py.on("error", (e) => console.error("[electron] no se pudo iniciar Python:", e));
}

function waitForServer(cb, tries = 0) {
  http
    .get(URL, (res) => { res.destroy(); cb(); })
    .on("error", () => {
      if (tries > 150) return cb();                            // ~22s max, abrimos igual
      setTimeout(() => waitForServer(cb, tries + 1), 150);
    });
}

function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 900,
    backgroundColor: "#0d0d0f",
    title: "Toribash Humanoid — Eval",
    autoHideMenuBar: true,
  });
  win.maximize();              // arranca MAXIMIZADA (con barra de titulo, no fullscreen sin bordes)
  win.loadURL(URL);
}

function killPy() {
  if (py) { try { py.kill(); } catch (e) {} py = null; }
}

app.whenReady().then(() => {
  startServer();
  waitForServer(createWindow);
});

app.on("window-all-closed", () => { killPy(); app.quit(); });
app.on("before-quit", killPy);
