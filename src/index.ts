import { spawn } from 'child_process';
import * as path from 'path';

export type CaptchaAction =
  | { action: 'click'; coordinates: [number, number] }
  | { action: 'drag'; source_coordinates: [number, number]; target_coordinates: [number, number] }
  | { action: 'type'; text: string; target_id?: number }
  | { action: 'wait'; duration_ms: number }
  | { action: 'request_updated_image' };

export interface SolveOptions {
  /** Backend for action planning: "ollama" or "openai" */
  planner?: 'ollama' | 'openai' | 'gemini';
  /** Model name for the planner */
  plannerModel?: string;
  /** Model name for attention extraction */
  attentionModel?: string;
  /** OpenAI API key (if using openai planner) */
  apiKey?: string;
  /** OpenAI base URL (if using openai planner) */
  apiBase?: string;
  /** Path to Python executable */
  pythonPath?: string;
}

/**
 * Solve a captcha using the CaptchaKraken Python backend.
 *
 * @param imagePath - Path to the captcha image
 * @param prompt - Context/instructions for solving
 * @param options - Solver configuration options
 * @returns A single CaptchaAction to perform
 */
export async function solveCaptcha(
  imagePath: string,
  prompt: string,
  options: SolveOptions = {}
): Promise<CaptchaAction> {
  return new Promise((resolve, reject) => {
    // Resolve path to CLI script relative to package root
    const pythonScript = path.resolve(__dirname, '../src/captchakraken/cli.py');

    const args = [pythonScript, imagePath, prompt];

    if (options.planner) {
      args.push('--planner', options.planner);
    }
    if (options.plannerModel) {
      args.push('--planner-model', options.plannerModel);
    }
    if (options.attentionModel) {
      args.push('--attention-model', options.attentionModel);
    }
    if (options.apiKey) {
      args.push('--api-key', options.apiKey);
    }
    if (options.apiBase) {
      args.push('--api-base', options.apiBase);
    }

    const pythonExec = options.pythonPath || 'python3';
    const proc = spawn(pythonExec, args);

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python script exited with code ${code}: ${stderr}`));
        return;
      }

      try {
        const result = JSON.parse(stdout);
        resolve(result);
      } catch {
        reject(new Error(`Failed to parse JSON output: ${stdout}`));
      }
    });
  });
}
