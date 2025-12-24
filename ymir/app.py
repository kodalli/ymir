"""FastAPI application for Ymir - Agentic Dataset Generator."""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocket

from ymir.api import templates, routers, lifespan

# Initialize FastAPI app
app = FastAPI(
    title="Ymir - Agentic Dataset Generator",
    description="Generate multi-turn tool-calling trajectories for agentic LLM training",
    lifespan=lifespan,
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="ymir/static"), name="static")
except RuntimeError:
    pass  # Already mounted during reload


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the main layout template."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


# Register all routers
for router in routers:
    app.include_router(router)


@app.websocket("/dev/reload")
async def dev_reload(websocket: WebSocket):
    """WebSocket endpoint for dev auto-reload. Browser refreshes when server restarts."""
    await websocket.accept()
    try:
        while True:
            await websocket.receive_text()
    except:
        pass


if __name__ == "__main__":
    uvicorn.run("ymir.app:app", host="0.0.0.0", port=8008, reload=True)
