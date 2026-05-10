#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(SCRIPT_DIR, "..");

function printUsage() {
  console.log(`Usage:
  node scripts/preprocess_gaussian_stream.mjs --output-dir <gs-sdf-output-dir> [options]
  node scripts/preprocess_gaussian_stream.mjs --input <gs.ply> [options]

Options:
  --output-dir <dir>         GS-SDF output directory that contains model/gs.ply
  --input <file>             Source Gaussian PLY file
  --variant <name>           Named runtime variant suffix (example: quality)
  --grid <x,y>               XY chunk grid size (default: 4,4)
  --max-sh <0|1|2|3>         Maximum spherical harmonics degree to keep (default: 1)
  --fractional-bits <int>    SPZ fractional bits override
  --opacity-threshold <f>    Drop splats with opacity below threshold
  --clip-min <x,y,z>         Optional clip min corner
  --clip-max <x,y,z>         Optional clip max corner
  --force                    Overwrite existing stream outputs
  --dry-run                  Print planned conversion without writing files
  --help                     Show this help
`);
}

function parseArgs(argv) {
  const options = {
    grid: [4, 4],
    maxSh: 1,
    variant: null,
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
      case "--variant":
        options.variant = parseVariant(argv[++index]);
        break;
      case "--grid":
        options.grid = parsePair(argv[++index], "--grid");
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

function parsePair(value, flag) {
  const parts = value.split(",").map((item) => Number.parseInt(item.trim(), 10));
  if (
    parts.length !== 2 ||
    parts.some((item) => !Number.isFinite(item) || item <= 0)
  ) {
    throw new Error(`${flag} expects positive integers x,y`);
  }
  return parts;
}

function parseVec3(value, flag) {
  const parts = value.split(",").map((item) => Number.parseFloat(item.trim()));
  if (parts.length !== 3 || parts.some((item) => Number.isNaN(item))) {
    throw new Error(`${flag} expects x,y,z`);
  }
  return parts;
}

function parseVariant(value) {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (!normalized || !/^[a-z0-9][a-z0-9_-]*$/.test(normalized)) {
    throw new Error("--variant expects letters, numbers, '-' or '_'");
  }
  return normalized;
}

async function resolvePaths(options) {
  let inputPath = options.input ? path.resolve(options.input) : null;
  let runName = null;

  if (options.outputDir) {
    const outputDir = path.resolve(options.outputDir);
    runName = path.basename(outputDir);
    inputPath ??= path.join(outputDir, "model", "gs.ply");
  }

  if (!inputPath) {
    throw new Error("Provide --output-dir or --input");
  }

  runName ??= path.parse(inputPath).name;
  const processedRoot = path.join(ROOT_DIR, "runtime", "processed", runName, "model");
  const variantSuffix = options.variant ? `_${options.variant}` : "";
  const chunkDir = path.join(processedRoot, `gs_chunks${variantSuffix}`);
  const metadataPath = path.join(processedRoot, `gs_chunks${variantSuffix}.json`);

  return {
    inputPath,
    processedRoot,
    chunkDir,
    metadataPath,
    runName,
    variant: options.variant ?? "balanced"
  };
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

async function importSparkModule() {
  globalThis.navigator ??= { xr: undefined, userAgent: "node" };
  globalThis.self ??= globalThis;
  return import("../examples/web-ui/node_modules/@sparkjsdev/spark/dist/spark.module.js");
}

function detectShDegree(headerText) {
  const restCount = headerText
    .split(/\r?\n/)
    .filter((line) => line.startsWith("property float f_rest_")).length;
  return restCount >= 45 ? 3 : restCount >= 24 ? 2 : restCount >= 9 ? 1 : 0;
}

function makeFilter(options) {
  return (center, opacity) => {
    if (typeof options.opacityThreshold === "number" && opacity < options.opacityThreshold) {
      return false;
    }
    if (options.clipMin && options.clipMax) {
      if (
        center.x < options.clipMin[0] ||
        center.y < options.clipMin[1] ||
        center.z < options.clipMin[2] ||
        center.x > options.clipMax[0] ||
        center.y > options.clipMax[1] ||
        center.z > options.clipMax[2]
      ) {
        return false;
      }
    }
    return true;
  };
}

function createEmptyBounds() {
  return {
    minX: Number.POSITIVE_INFINITY,
    minY: Number.POSITIVE_INFINITY,
    minZ: Number.POSITIVE_INFINITY,
    maxX: Number.NEGATIVE_INFINITY,
    maxY: Number.NEGATIVE_INFINITY,
    maxZ: Number.NEGATIVE_INFINITY
  };
}

function expandBounds(bounds, x, y, z) {
  bounds.minX = Math.min(bounds.minX, x);
  bounds.minY = Math.min(bounds.minY, y);
  bounds.minZ = Math.min(bounds.minZ, z);
  bounds.maxX = Math.max(bounds.maxX, x);
  bounds.maxY = Math.max(bounds.maxY, y);
  bounds.maxZ = Math.max(bounds.maxZ, z);
}

function isBoundsValid(bounds) {
  return Number.isFinite(bounds.minX) && Number.isFinite(bounds.maxX) && bounds.maxX >= bounds.minX;
}

function clampIndex(index, size) {
  return Math.min(Math.max(index, 0), size - 1);
}

function getChunkIndex(x, y, bounds, grid) {
  const sizeX = Math.max(bounds.maxX - bounds.minX, 1e-6);
  const sizeY = Math.max(bounds.maxY - bounds.minY, 1e-6);
  const cellX = clampIndex(Math.floor(((x - bounds.minX) / sizeX) * grid[0]), grid[0]);
  const cellY = clampIndex(Math.floor(((y - bounds.minY) / sizeY) * grid[1]), grid[1]);
  return cellY * grid[0] + cellX;
}

function chunkFileName(index) {
  return `chunk_${index.toString().padStart(3, "0")}.spz`;
}

async function removeDirIfExists(dirPath) {
  if (await fileExists(dirPath)) {
    await fs.rm(dirPath, { recursive: true, force: true });
  }
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

  const { inputPath, processedRoot, chunkDir, metadataPath, runName, variant } = await resolvePaths(options);
  if (!(await fileExists(inputPath))) {
    throw new Error(`Input file not found: ${inputPath}`);
  }

  if (!options.force && !options.dryRun && (await fileExists(metadataPath))) {
    throw new Error(`Chunk metadata already exists: ${metadataPath}. Re-run with --force to overwrite.`);
  }

  const inputStat = await fs.stat(inputPath);
  const spark = await importSparkModule();
  const fileBytes = new Uint8Array(await fs.readFile(inputPath));
  const reader = new spark.PlyReader({ fileBytes });
  await reader.parseHeader();
  const sourceShDegree = detectShDegree(reader.header);
  const outputShDegree = Math.min(sourceShDegree, options.maxSh);
  const filter = makeFilter(options);

  const acceptedBounds = createEmptyBounds();
  let acceptedSplats = 0;
  reader.parseSplats((index, x, y, z, _sx, _sy, _sz, _qx, _qy, _qz, _qw, opacity) => {
    if (!filter({ x, y, z }, opacity)) {
      return;
    }
    acceptedSplats += 1;
    expandBounds(acceptedBounds, x, y, z);
  });

  if (!acceptedSplats || !isBoundsValid(acceptedBounds)) {
    throw new Error("No splats survived the requested clip / opacity filter.");
  }

  const plan = {
    stage: "plan",
    runName,
    inputPath,
    inputSize: formatBytes(inputStat.size),
    inputBytes: inputStat.size,
    sourceSplats: reader.numSplats,
    acceptedSplats,
    sourceShDegree,
    outputShDegree,
    grid: {
      x: options.grid[0],
      y: options.grid[1],
      total: options.grid[0] * options.grid[1]
    },
    variant,
    chunkDir,
    metadataPath
  };
  console.log(JSON.stringify(plan, null, 2));

  if (options.dryRun) {
    return;
  }

  const counts = new Array(options.grid[0] * options.grid[1]).fill(0);
  const chunkBounds = counts.map(() => createEmptyBounds());
  reader.parseSplats((index, x, y, z, _sx, _sy, _sz, _qx, _qy, _qz, _qw, opacity) => {
    if (!filter({ x, y, z }, opacity)) {
      return;
    }
    const chunkIndex = getChunkIndex(x, y, acceptedBounds, options.grid);
    counts[chunkIndex] += 1;
    expandBounds(chunkBounds[chunkIndex], x, y, z);
  });

  await fs.mkdir(processedRoot, { recursive: true });
  await removeDirIfExists(chunkDir);
  await fs.mkdir(chunkDir, { recursive: true });

  const writers = counts.map((count) => {
    if (count <= 0) {
      return null;
    }
    return new spark.SpzWriter({
      numSplats: count,
      shDegree: outputShDegree,
      fractionalBits: options.fractionalBits,
      flagAntiAlias: true
    });
  });

  const writeIndices = counts.map(() => 0);
  const lastChunkIndex = new Int32Array(reader.numSplats);
  const lastLocalIndex = new Int32Array(reader.numSplats);
  lastChunkIndex.fill(-1);
  lastLocalIndex.fill(-1);

  reader.parseSplats(
    (index, x, y, z, scaleX, scaleY, scaleZ, quatX, quatY, quatZ, quatW, opacity, r, g, b) => {
      if (!filter({ x, y, z }, opacity)) {
        return;
      }
      const chunkIndex = getChunkIndex(x, y, acceptedBounds, options.grid);
      const writer = writers[chunkIndex];
      if (!writer) {
        return;
      }
      const localIndex = writeIndices[chunkIndex];
      writeIndices[chunkIndex] += 1;
      lastChunkIndex[index] = chunkIndex;
      lastLocalIndex[index] = localIndex;
      writer.setCenter(localIndex, x, y, z);
      writer.setScale(localIndex, scaleX, scaleY, scaleZ);
      writer.setQuat(localIndex, quatX, quatY, quatZ, quatW);
      writer.setAlpha(localIndex, opacity);
      writer.setRgb(localIndex, r, g, b);
    },
    (index, sh1, sh2, sh3) => {
      if (outputShDegree <= 0) {
        return;
      }
      const chunkIndex = lastChunkIndex[index];
      const localIndex = lastLocalIndex[index];
      if (chunkIndex < 0 || localIndex < 0) {
        return;
      }
      const writer = writers[chunkIndex];
      if (!writer) {
        return;
      }
      writer.setSh(
        localIndex,
        sh1,
        outputShDegree >= 2 ? sh2 : undefined,
        outputShDegree >= 3 ? sh3 : undefined
      );
    }
  );

  const startedAt = Date.now();
  const chunkMetadata = [];
  let totalOutputBytes = 0;

  for (let index = 0; index < writers.length; index += 1) {
    const writer = writers[index];
    const count = counts[index];
    if (!writer || count <= 0) {
      continue;
    }

    const fileName = chunkFileName(index);
    const targetPath = path.join(chunkDir, fileName);
    const tempPath = `${targetPath}.tmp`;
    const spzBytes = await writer.finalize();
    await fs.writeFile(tempPath, spzBytes);
    await fs.rename(tempPath, targetPath);

    const bounds = chunkBounds[index];
    totalOutputBytes += spzBytes.byteLength;
    chunkMetadata.push({
      id: `chunk-${index}`,
      file: fileName,
      path: targetPath,
      bytes: spzBytes.byteLength,
      splats: count,
      center: {
        x: (bounds.minX + bounds.maxX) * 0.5,
        y: (bounds.minY + bounds.maxY) * 0.5,
        z: (bounds.minZ + bounds.maxZ) * 0.5
      },
      bounds: {
        min: { x: bounds.minX, y: bounds.minY, z: bounds.minZ },
        max: { x: bounds.maxX, y: bounds.maxY, z: bounds.maxZ }
      }
    });
  }

  const elapsedMs = Date.now() - startedAt;
  const metadata = {
    format: "gs-chunks",
    source: {
      path: inputPath,
      bytes: inputStat.size,
      splats: reader.numSplats,
      shDegree: sourceShDegree
    },
    output: {
      root: chunkDir,
      bytes: totalOutputBytes,
      splats: acceptedSplats,
      shDegree: outputShDegree,
      chunkCount: chunkMetadata.length,
      grid: { x: options.grid[0], y: options.grid[1] }
    },
    variant,
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
      compressionRatio: inputStat.size / Math.max(totalOutputBytes, 1)
    },
    chunks: chunkMetadata,
    createdAt: new Date().toISOString()
  };

  await fs.writeFile(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);

  console.log(
    JSON.stringify(
      {
        stage: "done",
        metadataPath,
        chunkDir,
        chunkCount: chunkMetadata.length,
        outputBytes: totalOutputBytes,
        outputSize: formatBytes(totalOutputBytes),
        compressionRatio: Number(metadata.stats.compressionRatio.toFixed(3)),
        elapsedSeconds: Number((elapsedMs / 1000).toFixed(1))
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
