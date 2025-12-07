import * as path from 'path';
import * as dotenv from 'dotenv';
import { spawn } from 'child_process';
import * as fs from 'fs';

const envPath = path.resolve(__dirname, '../../../.env');
console.log('Loading .env from:', envPath);
console.log('File exists:', fs.existsSync(envPath));

const result = dotenv.config({ path: envPath });
if (result.error) {
  console.error('Dotenv error:', result.error);
}

console.log('GEMINI_API_KEY:', process.env.GEMINI_API_KEY ? 'Set' : 'Not Set');

const pythonPath = path.resolve(__dirname, '../../../venv/bin/python');
console.log('Python Path:', pythonPath);
console.log('Python exists:', fs.existsSync(pythonPath));

const pythonProcess = spawn(pythonPath, ['-c', 'import PIL; print("PIL Version:", PIL.__version__)']);

pythonProcess.stdout.on('data', (data) => {
  console.log(`Python stdout: ${data}`);
});

pythonProcess.stderr.on('data', (data) => {
  console.error(`Python stderr: ${data}`);
});

pythonProcess.on('close', (code) => {
  console.log(`Python exited with code ${code}`);
});

