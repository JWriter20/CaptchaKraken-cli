import { spawn } from 'child_process';
import * as path from 'path';

export type CaptchaAction =
  | {
    action: 'click';
    coordinates?: [number, number];
    all_coordinates?: [number, number][];
    target_ids?: number[];
    bounding_boxes?: number[][];
  }
  | {
    action: 'drag';
    source_coordinates: [number, number];
    target_coordinates: [number, number];
    source_id?: number;
    target_id?: number;
  }
  | { action: 'type'; text: string; target_id?: number }
  | { action: 'wait'; duration_ms: number }
  | { action: 'request_updated_image' }
  | { action: 'verify'; target_id?: number };

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
  /** Extracted prompt text */
  promptText?: string;
  /** Extracted prompt image URL */
  promptImageUrl?: string;
  /** Selector for the challenge element */
  challengeElementSelector?: string;
  /** Detected interactable elements */
  elements?: any[];
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
    if (options.promptText) {
      args.push('--prompt-text', options.promptText);
    }
    if (options.promptImageUrl) {
      args.push('--prompt-image-url', options.promptImageUrl);
    }
    if (options.challengeElementSelector) {
      args.push('--challenge-element-selector', options.challengeElementSelector);
    }
    if (options.elements) {
      args.push('--elements', JSON.stringify(options.elements));
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

/**
 * Apply bounding box overlays to an image using Python script.
 * 
 * @param imagePath - Path to the image file (will be modified in-place)
 * @param boxes - Array of boxes {text, bbox: [x, y, w, h]}
 * @param pythonPath - Path to python executable (optional)
 */
export async function applyOverlays(
  imagePath: string,
  boxes: Array<{ text: string, bbox: [number, number, number, number] }>,
  pythonPath: string = 'python3'
): Promise<void> {
  if (!boxes || boxes.length === 0) return;

  return new Promise((resolve, reject) => {
    const pythonScript = path.resolve(__dirname, '../src/captchakraken/overlay.py');
    const boxesJson = JSON.stringify(boxes);

    const proc = spawn(pythonPath, [pythonScript, imagePath, boxesJson]);

    let stderr = '';

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Overlay script exited with code ${code}: ${stderr}`));
        return;
      }
      resolve();
    });
  });
}
