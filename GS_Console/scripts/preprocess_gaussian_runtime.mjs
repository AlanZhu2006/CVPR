#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(SCRIPT_DIR, "..");

function printUsage() {
  console.log(`Usage:
  node scripts/preprocess_gaussian_runtime.mjs --output-dir <gs-sdf-output-dir> [options]
  node scripts/preprocess_gaussian_runtime.mjs --input <gs.ply> [--output <gs_runtime.spz>] [options]

Options:
  --output-dir <dir>         GS-SDF output directory that contains model/gs.ply
  --input <file>             Source Gaussian PLY file
  --output <file>            Output runtime SPZ path
  --metadata <file>          Metadata JSON path (default: alongside output as gs_runtime.json)
  --max-sh <0|1|2|3>         Maximum spherical harmonics degree to keep (default: 1)
  --fractional-bits <int>    SPZ fractional bits override
  --opacity-threshold <f>    Drop splats with opacity below threshold
  --clip-min <x,y,z>         Optional clip min corner
  --clip-max <x,y,z>         Optional clip max corner
  --force                    Overwrite existing output
  --dry-run                  Print planned conversion without writing files
  --help                     Show this help
`);
}

function parseArgs(argv) {
  const options = {
    maxSh: 1,
    force: false,
    dryRun: false
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--output-dir":
        options.outputDir = argv[++index];
        break;
      case "--input":
        options.input = argv[++index];
        break;
      case "--output":
        options.output = argv[++index];
        break;
      case "--metadata":
        options.metadata = argv[++index];
        break;
      case "--max-sh":
        options.maxSh = Number.parseInt(argv[++index], 10);
        break;
      case "--fractional-bits":
        options.fractionalBits = Number.parseInt(argv[++index], 10);
        break;
      case "--opacity-threshold":
        options.opacityThreshold = Number.parseFloat(argv[++index]);
        break;
      case "--clip-min":
        options.clipMin = parseVec3(argv[++index], "--clip-min");
        break;
      case "--clip-max":
        options.clipMax = parseVec3(argv[++index], "--clip-max");
        break;
      case "--force":
        options.force = true;
        break;
      case "--dry-run":
        options.dryRun = true;
        break;
      case "--help":
        options.help = true;
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function parseVec3(value, flag) {
  const parts = value.split(",").map((item) => Number.parseFloat(item.trim()));
  if (parts.length !== 3 || parts.some((item) => Number.isNaN(item))) {
    throw new Error(`${flag} expects x,y,z`);
  }
  return parts;
}

async function resolvePaths(options) {
  let inputPath = options.input ? path.resolve(options.input) : null;
  let outputPath = options.output ? path.resolve(options.output) : null;
  let metadataPath = options.metadata ? path.resolve(options.metadata) : null;

  if (options.outputDir) {
    const outputDir = path.resolve(options.outputDir);
    inputPath ??= path.join(outputDir, "model", "gs.ply");
    const processedRoot = path.join(ROOT_DIR, "runtime", "processed", path.basename(outputDir), "model");
    outputPath ??= path.join(processedRoot, "gs_runtime.spz");
    metadataPath ??= path.join(processedRoot, "gs_runtime.json");
  }

  if (!inputPath) {
    throw new Error("Provide --output-dir or --input");
  }

  if (!outputPath) {
    const parsed = path.parse(inputPath);
    outputPath = path.join(parsed.dir, `${parsed.name}_runtime.spz`);
  }

  metadataPath ??= path.join(path.dirname(outputPath), "gs_runtime.json");

  return {
    inputPath,
    outputPath,
    metadataPath
  };
}

async function parsePlyHeader(filePath) {
  const handle = await fs.open(filePath, "r");
  try {
    const chunkSize = 64 * 1024;
    let buffer = Buffer.alloc(0);
    let position = 0;
    while (true) {
      const chunk = Buffer.alloc(chunkSize);
      const { bytesRead } = await handle.read(chunk, 0, chunkSize, position);
      if (bytesRead <= 0) {
        throw new Error(`Failed to parse PLY header from ${filePath}`);
      }
      buffer = Buffer.concat([buffer, chunk.subarray(0, bytesRead)]);
      position += bytesRead;
      const markerIndex = buffer.indexOf("end_header\n");
      if (markerIndex >= 0) {
        const headerText = buffer.subarray(0, markerIndex + "end_header\n".length).toString("latin1");
        const lines = headerText.split(/\r?\n/).filter(Boolean);
        const vertexLine = lines.find((line) => line.startsWith("element vertex "));
        const vertexCount = vertexLine ? Number.parseInt(vertexLine.split(/\s+/)[2], 10) : null;
        const restCount = lines.filter((line) => line.startsWith("property float f_rest_")).length;
        const shDegree = restCount >= 45 ? 3 : restCount >= 24 ? 2 : restCount >= 9 ? 1 : 0;
        return {
          headerText,
          vertexCount,
          shDegree
        };
      }
      if (buffer.length > 4 * 1024 * 1024) {
        throw new Error(`PLY header in ${filePath} is unexpectedly large`);
      }
    }
  } finally {
    await handle.close();
  }
}

async function importSparkModule() {
  globalThis.navigator ??= { xr: undefined, userAgent: "node" };
  globalThis.self ??= globalThis;
  return import("../examples/web-ui/node_modules/@sparkjsdev/spark/dist/spark.module.js");
}

function buildTranscodeInput(fileBytes, inputPath, options) {
  const transcodeInput = {
    inputs: [
      {
        fileBytes,
        fileType: "ply",
        pathOrUrl: inputPath
      }
    ],
    maxSh: options.maxSh
  };

  if (typeof options.fractionalBits === "number") {
    transcodeInput.fractionalBits = options.fractionalBits;
  }

  if (typeof options.opacityThreshold === "number") {
    transcodeInput.opacityThreshold = options.opacityThreshold;
  }

  if (options.clipMin || options.clipMax) {
    if (!options.clipMin || !options.clipMax) {
      throw new Error("Provide both --clip-min and --clip-max");
    }
    transcodeInput.clipXyz = {
      min: options.clipMin,
      max: options.clipMax
    };
  }

  return transcodeInput;
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function formatBytes(bytes) {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printUsage();
    return;
  }

  if (![0, 1, 2, 3].includes(options.maxSh)) {
    throw new Error("--max-sh must be 0, 1, 2, or 3");
  }

  const { inputPath, outputPath, metadataPath } = await resolvePaths(options);
  const inputExists = await fileExists(inputPath);
  if (!inputExists) {
    throw new Error(`Input file not found: ${inputPath}`);
  }

  if (!options.force && !options.dryRun && (await fileExists(outputPath))) {
    throw new Error(`Output already exists: ${outputPath}. Re-run with --force to overwrite.`);
  }

  const inputStat = await fs.stat(inputPath);
  const header = await parsePlyHeader(inputPath);
  const summary = {
    inputPath,
    outputPath,
    metadataPath,
    inputBytes: inputStat.size,
    inputSize: formatBytes(inputStat.size),
    sourceSplats: header.vertexCount,
    sourceShDegree: header.shDegree,
    targetMaxSh: options.maxSh,
    fractionalBits: options.fractionalBits ?? null,
    opacityThreshold: options.opacityThreshold ?? null,
    clipXyz:
      options.clipMin && options.clipMax
        ? { min: options.clipMin, max: options.clipMax }
        : null
  };

  console.log(JSON.stringify({ stage: "plan", ...summary }, null, 2));

  if (options.dryRun) {
    return;
  }

  const spark = await importSparkModule();
  const fileBytes = new Uint8Array(await fs.readFile(inputPath));
  const transcodeInput = buildTranscodeInput(fileBytes, inputPath, options);

  const startedAt = Date.now();
  const { fileBytes: spzBytes } = await spark.transcodeSpz(transcodeInput);
  const elapsedMs = Date.now() - startedAt;

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const tempOutputPath = `${outputPath}.tmp`;
  await fs.writeFile(tempOutputPath, spzBytes);
  await fs.rename(tempOutputPath, outputPath);

  const reader = new spark.SpzReader({ fileBytes: spzBytes });
  await reader.parseHeader();

  const outputStat = await fs.stat(outputPath);
  const metadata = {
    format: "spz",
    source: {
      path: inputPath,
      bytes: inputStat.size,
      splats: header.vertexCount,
      shDegree: header.shDegree
    },
    output: {
      path: outputPath,
      bytes: outputStat.size,
      splats: reader.numSplats,
      shDegree: reader.shDegree
    },
    options: {
      maxSh: options.maxSh,
      fractionalBits: options.fractionalBits ?? null,
      opacityThreshold: options.opacityThreshold ?? null,
      clipXyz:
        options.clipMin && options.clipMax
          ? {
              min: options.clipMin,
              max: options.clipMax
            }
          : null
    },
    stats: {
      elapsedMs,
      compressionRatio: inputStat.size / Math.max(outputStat.size, 1)
    },
    createdAt: new Date().toISOString()
  };

  await fs.writeFile(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);

  console.log(
    JSON.stringify(
      {
        stage: "done",
        outputPath,
        outputBytes: outputStat.size,
        outputSize: formatBytes(outputStat.size),
        compressionRatio: Number(metadata.stats.compressionRatio.toFixed(3)),
        elapsedSeconds: Number((elapsedMs / 1000).toFixed(1)),
        outputSplats: reader.numSplats,
        outputShDegree: reader.shDegree,
        metadataPath
      },
      null,
      2
    )
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
