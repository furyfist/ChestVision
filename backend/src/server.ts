// backend/src/server.ts

import express, { Request, Response } from 'express';
import multer from 'multer';
import axios from 'axios';
import cors from 'cors';
import FormData from 'form-data';

const app = express();
const port = 5000; // The port for this Node.js backend server

// --- 1. Middleware Setup ---

// Enable CORS: This is crucial for allowing your React app (on a different port)
// to send requests to this backend server without being blocked by browser security.
app.use(cors());

// Configure Multer: This middleware handles multipart/form-data, which is
// used for uploading files. We tell it to store the uploaded file in memory.
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// --- 2. Define API Route ---

// The URL of your running Python AI service
const AI_SERVICE_URL = 'http://localhost:8000/predict';

// Define a POST route at '/api/upload'.
// The 'upload.single('file')' part tells Multer to expect one file named 'file'.
app.post('/api/upload', upload.single('file'), async (req: Request, res: Response) => {
  // Check if a file was actually uploaded.
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded.' });
  }

  console.log(`File received: ${req.file.originalname}, Size: ${req.file.size} bytes`);
  console.log('Forwarding to Python AI service...');

  try {
    // We must forward the image to the Python service as 'multipart/form-data'.
    // The 'form-data' library helps us build this request.
    const formData = new FormData();

    // Append the file's data (buffer) to the form. The key 'file' must match
    // what the Flask app expects: `request.files['file']`.
    formData.append('file', req.file.buffer, req.file.originalname);

    // Use Axios to make the POST request to the Python service.
    const response = await axios.post(AI_SERVICE_URL, formData, {
      headers: {
        ...formData.getHeaders(), // Important: This sets the correct 'Content-Type' header.
      },
    });

    console.log('Prediction received from AI service:', response.data);

    // Send the prediction from the AI service back to the frontend.
    res.json(response.data);

  } catch (error: any) {
    // If the Python service is down or there's an error, catch it.
    console.error('Error forwarding request to AI service:', error.message);
    res.status(500).json({ error: 'Failed to get prediction from AI service.' });
  }
});

// --- 3. Start the Server ---

// Start listening for incoming requests on the specified port.
app.listen(port, () => {
  console.log(`Backend server is running at http://localhost:${port}`);
});