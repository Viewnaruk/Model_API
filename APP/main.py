from fastapi import FastAPI
import uvicorn
import os

# สร้าง instance ของ FastAPI
app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Hello, World'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 9000)))