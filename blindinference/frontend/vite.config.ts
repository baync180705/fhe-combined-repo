import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {loadEnv} from 'vite';
import type { ConfigEnv, PluginOption, UserConfig } from 'vite';

async function getOptionalWasmPlugin(): Promise<PluginOption[]> {
  try {
    const wasm = await import('vite-plugin-wasm');
    return [wasm.default()];
  } catch {
    return [];
  }
}

export default async function config({mode}: ConfigEnv): Promise<UserConfig> {
  const env = loadEnv(mode, '.', '');
  const wasmPlugins = await getOptionalWasmPlugin();
  return {
    plugins: [react(), ...wasmPlugins, tailwindcss()],
    optimizeDeps: {
      include: ['iframe-shared-storage', 'tweetnacl'],
      exclude: ['@cofhe/sdk', 'tfhe', 'fhenixjs'],
    },
    build: {
      target: 'esnext',
      commonjsOptions: {
        include: [/iframe-shared-storage/, /tweetnacl/, /node_modules/],
      },
    },
    define: {
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
        'fhenixjs-access-control': path.resolve(
          __dirname,
          'node_modules/fhenixjs/lib/commonjs/extensions/access_control/index.js',
        ),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modifyâfile watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
    },
    worker: {
      format: 'es',
      plugins: () => wasmPlugins,
    },
  };
}
