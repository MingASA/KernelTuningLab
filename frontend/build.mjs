import { build } from "esbuild";
import { readFile, writeFile, mkdir } from "node:fs/promises";
const out = new URL("../kernel_tuning_lab/console_assets/", import.meta.url);
await mkdir(out, { recursive: true });
await build({
  entryPoints: ["src/app.tsx"],
  bundle: true,
  minify: true,
  target: "es2022",
  outfile: new URL("app.js", out).pathname,
  jsx: "automatic",
  define: { "process.env.NODE_ENV": '"production"' },
  legalComments: "none",
  logLevel: "info",
});
const licenses = [];
for (const name of [
  "react",
  "react-dom",
  "scheduler",
  "echarts",
  "zrender",
  "tslib",
]) {
  const pkg = JSON.parse(
    await readFile(
      new URL(`./node_modules/${name}/package.json`, import.meta.url),
      "utf8",
    ),
  );
  for (const candidate of [
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "CopyrightNotice.txt",
    "NOTICE",
  ]) {
    try {
      licenses.push(
        `${name} ${pkg.version}\n${await readFile(new URL(`./node_modules/${name}/${candidate}`, import.meta.url), "utf8")}\n`,
      );
    } catch (e) {
      if (e.code !== "ENOENT") throw e;
    }
  }
}
await writeFile(
  new URL("THIRD_PARTY_LICENSES.txt", out),
  licenses.join("\n--------------------------------\n").replace(/\r\n/g, "\n").trimEnd() + "\n",
);
