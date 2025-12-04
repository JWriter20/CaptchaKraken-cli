import { spawn } from 'child_process';
import * as path from 'path';

export type CaptchaAction =
  | { action: 'click', target_id?: number, coordinates?: [number, number] }
  | { action: 'drag', source_id?: number, source_coordinates?: [number, number], target_id?: number, target_coordinates?: [number, number] }
  | { action: 'type', text: string, target_id?: number }
  | { action: 'wait', duration_ms: number }
  | { action: 'request_updated_image' };

export interface SolveOptions {
  strategy?: 'omniparser' | 'holo2';
  apiKey?: string;
  pythonPath?: string;
}

export async function solveCaptcha(imagePath: string, prompt: string, options: SolveOptions = {}): Promise<CaptchaAction[]> {
  return new Promise((resolve, reject) => {
    // Assuming the python script is in src/captchakraken/cli.py relative to the package root
    // If this runs from dist/index.js, we need to go up one level to root, then into src/captchakraken
    // But if we are in src (dev), it's different.
    // Let's assume the package structure preserves src/captchakraken or we point to it.
    // For this implementation, we'll try to resolve it relative to the package root.

    // This logic depends on where the file ends up. 
    // If we compile to dist/, we expect src/captchakraken to be available or we should bundle it.
    // For now, we point to the source file.
    const pythonScript = path.resolve(__dirname, '../src/captchakraken/cli.py');

    const args = [pythonScript, imagePath, prompt];
    if (options.strategy) {
      args.push('--strategy', options.strategy);
    }
    if (options.apiKey) {
      args.push('--api_key', options.apiKey);
    }

    const pythonExec = options.pythonPath || 'python3';

    const process = spawn(pythonExec, args);

    let stdout = '';
    let stderr = '';

    process.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    process.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    process.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python script exited with code ${code}: ${stderr}`));
        return;
      }

      try {
        const result = JSON.parse(stdout);
        resolve(result);
      } catch (e) {
        reject(new Error(`Failed to parse JSON output: ${stdout}`));
      }
    });
  });
}

