const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const PYTHON_SERVICE_URL = "http://127.0.0.1:8000";

async function testUpload() {
  try {
    const filePath = path.join(__dirname, 'test.txt');
    fs.writeFileSync(filePath, 'dummy audio content');

    const audioBuffer = fs.readFileSync(filePath);
    
    console.log(`Sending buffer of size: ${audioBuffer.byteLength}`);

    const formData = new FormData();
    formData.append('user_id', 'test_user_from_script');
    formData.append('reference_audio', audioBuffer, { 
      filename: 'test.wav',
      contentType: 'audio/wav' 
    });

    const response = await axios.post(
      `${PYTHON_SERVICE_URL}/audio/register`,
      formData,
      {
        headers: { ...formData.getHeaders() },
        timeout: 120000,
      }
    );

    console.log('✅ Success:', response.data);
  } catch (err) {
    if (err.response) {
      console.error('❌ Failed (Response):', err.response.status, err.response.data);
    } else {
      console.error('❌ Failed (Network/Other):', err.message);
    }
  }
}

testUpload();
