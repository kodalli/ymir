import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ymir.routes import templates, routers, lifespan

# Initialize global variables
app = FastAPI(title="Ymir AI Dataset Tools", lifespan=lifespan)


# Create static files directory for CSS and JS
try:
    app.mount("/static", StaticFiles(directory="ymir/static"), name="static")
except RuntimeError:
    # This happens when reloading the app (files already mounted)
    pass


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the main layout template only (content will be loaded via HTMX)"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )


for router in routers:
    app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("ymir.app:app", host="0.0.0.0", port=8008, reload=True)
