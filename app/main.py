from fastapi import FastAPI, Request  # Add Request import
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from app.routers import documents

# Create FastAPI app
app = FastAPI(
    title="Legal Document Simplifier",
    description="Simplify complex legal documents for easy understanding",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router, prefix="/api", tags=["Documents"])

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="app/templates")

# Root endpoint to serve the HTML interface
@app.get("/", include_in_schema=False)
async def root(request: Request):  # Add the Request parameter
    return templates.TemplateResponse("index.html", {"request": request})