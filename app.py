import os
from waitress import serve
from printlab.server import create_app

app = create_app()

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=int(os.environ.get('PORT', '18923')), threads=4, max_request_body_size=25 * 1024 * 1024)
