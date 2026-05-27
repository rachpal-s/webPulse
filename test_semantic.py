import asyncio, sys
sys.path.insert(0,'.')
from rag.ollama import get_ollama_client
async def test():
    ollama = get_ollama_client()
    emb = await ollama.embed(['test'])
    print('Embedding dim:', len(emb[0]))
asyncio.run(test())