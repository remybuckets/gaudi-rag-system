# Frontend (Next.js)

Scaffold this with the official tool rather than by hand:

```bash
# from the repo root
npx create-next-app@latest frontend --ts --app --eslint
```

Then build two things against the backend at http://localhost:8000:
1. **Upload** — POST a PDF to `/upload`.
2. **Chat** — POST a question to `/chat` and render the streamed response
   (read the response body as a stream; append deltas as they arrive).

Set `NEXT_PUBLIC_API_URL` in `frontend/.env.local`. Deploy on Vercel.
