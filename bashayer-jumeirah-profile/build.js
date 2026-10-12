#!/usr/bin/env node
// Renders index.html to Bashayer_Qaid_Jumeirah_Hospitality_Profile.pdf
// and runs layout checks (overflow, page count, unfinished fields).
//
//   node build.js            build the PDF
//   node build.js --preview  also write PNG previews of each page to ./preview

const path = require("path");
const fs = require("fs");
const { execFileSync } = require("child_process");
const { chromium } = require("playwright");

const ROOT = __dirname;
const OUT = path.join(ROOT, "Bashayer_Qaid_Jumeirah_Hospitality_Profile.pdf");
const EXPECTED_PAGES = 6;

function launchOptions() {
  // Use a preinstalled Chromium when one is available (e.g. cloud containers).
  const candidates = [process.env.CHROMIUM_PATH, "/opt/pw-browsers/chromium"].filter(Boolean);
  for (const p of candidates) {
    try {
      const stat = fs.statSync(p);
      if (stat.isFile()) return { executablePath: p };
      const bin = fs.readdirSync(p).map((f) => path.join(p, f)).find((f) => /chrom/.test(f) && fs.statSync(f).isFile());
      if (bin) return { executablePath: bin };
    } catch (_) { /* fall through to Playwright's default browser */ }
  }
  return {};
}

(async () => {
  const browser = await chromium.launch(launchOptions());
  const page = await browser.newPage();
  await page.goto("file://" + path.join(ROOT, "index.html"), { waitUntil: "load" });
  await page.evaluate(() => document.fonts.ready);
  await page.emulateMedia({ media: "print" });

  const report = await page.evaluate(() => {
    const issues = [];
    const pages = [...document.querySelectorAll(".page")];
    pages.forEach((pg, i) => {
      const box = pg.getBoundingClientRect();
      const folio = pg.querySelector(".folio");
      const limit = folio ? folio.getBoundingClientRect().top - 2 : box.bottom;
      for (const el of pg.querySelectorAll("p, li, dd, dt, h1, h2, h3, .row, .contact, .areas, .name-block")) {
        if (el.closest(".folio")) continue;
        const r = el.getBoundingClientRect();
        if (r.bottom > limit + 0.5 || r.right > box.right + 0.5) {
          issues.push(`page ${i + 1}: "${el.textContent.trim().slice(0, 50)}…" overflows the page`);
        }
      }
    });
    const fonts = [...document.fonts].filter((f) => f.status === "loaded").map((f) => `${f.family} ${f.weight} ${f.style}`);
    const todo = [...document.querySelectorAll(".to-complete")].map((el) => el.textContent.trim());
    return { pageCount: pages.length, issues, fonts, todo };
  });

  await page.pdf({ path: OUT, preferCSSPageSize: true, printBackground: true, tagged: true, outline: false });
  await browser.close();

  console.log(`PDF written: ${OUT}`);
  console.log(`Pages in source: ${report.pageCount} (expected ${EXPECTED_PAGES})`);
  console.log(`Fonts loaded: ${report.fonts.length}`);
  if (report.issues.length) {
    console.log("\nLAYOUT PROBLEMS:");
    report.issues.forEach((m) => console.log("  - " + m));
  } else {
    console.log("Layout check: no overflow detected.");
  }
  if (report.todo.length) {
    console.log("\nNOT SEND-READY — fields still to complete:");
    report.todo.forEach((m) => console.log("  - " + m));
  } else {
    console.log("All fields completed.");
  }

  if (process.argv.includes("--preview")) {
    const dir = path.join(ROOT, "preview");
    fs.mkdirSync(dir, { recursive: true });
    try {
      execFileSync("pdftoppm", ["-png", "-r", "80", OUT, path.join(dir, "page")]);
      console.log(`\nPreviews written to ${dir}`);
    } catch (e) {
      console.log("\n(pdftoppm not found — skipping PNG previews)");
    }
  }
  process.exitCode = report.issues.length || report.pageCount !== EXPECTED_PAGES ? 1 : 0;
})();
