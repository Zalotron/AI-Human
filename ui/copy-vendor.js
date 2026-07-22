// Copia Three.js (build + OrbitControls) desde node_modules a ./vendor,
// que es lo que sirve server.py. Se corre una vez despues de `npm install`.
const fs = require("fs");
const path = require("path");

const TREE = path.join(__dirname, "node_modules", "three");
const VEN = path.join(__dirname, "vendor");

const files = [
  ["build/three.module.js", "three.module.js"],
  ["examples/jsm/controls/OrbitControls.js", "addons/controls/OrbitControls.js"],
  ["examples/jsm/loaders/GLTFLoader.js", "addons/loaders/GLTFLoader.js"],
  ["examples/jsm/utils/BufferGeometryUtils.js", "addons/utils/BufferGeometryUtils.js"],
];

for (const [src, dst] of files) {
  const destPath = path.join(VEN, dst);
  fs.mkdirSync(path.dirname(destPath), { recursive: true });   // crear la carpeta destino
  fs.copyFileSync(path.join(TREE, src), destPath);
  console.log("copiado ->", dst);
}
console.log("vendor listo");
