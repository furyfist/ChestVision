import express, { Request, Response } from 'express';
import multer from 'multer';
import axios from 'axios';
import cors from 'cors';
import FormData from 'form-data';

const app = express();
const port = 5000;

app.use(cors());

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

const AI_SERVICE_URL = 'http://localhost:8000/predict';

app.post('/api/upload', upload.single('file'), async (req: Request, res: Response) => {

  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded.' });
  }

  console.log(`File received: ${req.file.originalname}, Size: ${req.file.size} bytes`);
  console.log('Forwarding to Python AI service...');

  try {

    const formData = new FormData();

    formData.append('file', req.file.buffer, req.file.originalname);

    const response = await axios.post(AI_SERVICE_URL, formData, {
      headers: {
        ...formData.getHeaders(), 
      },
    });

    console.log('Prediction received from AI service:', response.data);

    res.json(response.data);

  } catch (error: any) {

    console.error('Error forwarding request to AI service:', error.message);
    res.status(500).json({ error: 'Failed to get prediction from AI service.' });
  }
});


app.listen(port, () => {
  console.log(`Backend server is running at http://localhost:${port}`);
});