import httpx, asyncio, sys
sys.path.insert(0,'.')
from config import get_settings
cfg = get_settings()

async def check():
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post('https://api.jina.ai/v1/embeddings',
            headers={'Authorization': f'Bearer {cfg.jina_api_key}',
                     'Content-Type': 'application/json'},
            json={'model': 'jina-embeddings-v3',
                  'input': ['test'],
                  'task': 'retrieval.passage'})
        print('Status:', r.status_code)
        print('Headers:', dict(r.headers))
        print('Body:', r.text[:200])

asyncio.run(check())