import uvicorn ##ASGI
from fastapi import FastAPI
import os

# 2. Create the app object
app = FastAPI()

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 9000)))


