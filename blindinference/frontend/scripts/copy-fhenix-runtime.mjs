import { cp, mkdir, rm } from 'node:fs/promises';
import path from 'node:path';

const rootDir = process.cwd();
const sourceDir = path.join(rootDir, 'node_modules', 'fhenixjs', 'dist');
const targetDir = path.join(rootDir, 'public', 'vendor', 'fhenixjs');

async function main() {
  await mkdir(path.dirname(targetDir), { recursive: true });
  await rm(targetDir, { recursive: true, force: true });
  await cp(sourceDir, targetDir, { recursive: true });
  console.log(`Copied fhenixjs runtime from ${sourceDir} to ${targetDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
