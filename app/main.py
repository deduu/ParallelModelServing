# app/main.py
from fastapi import FastAPI
from fastapi.requests import Request
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from .utils.lifespan import lifespan
from .utils.logging_config import setup_logging
from .api import generate, status

# Setup logging
logger = setup_logging()

# Create FastAPI instance with lifespan
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(generate.router)
app.include_router(status.router)

# Root endpoint (optional)
@app.get("/")
async def root():
    return {"message": "LLM API is running."}

@app.post("/process/")
async def long_running_process(request: Request):
    try:
        # Simulate a long-running process
        for i in range(10):
            if await request.is_disconnected():
                logger.info("Client disconnected, cancelling task...")
                raise asyncio.CancelledError()
            await asyncio.sleep(1)  # Simulate work
            logger.info(f"Processing step {i+1}/10")
        
        return {"status": "completed", "message": "Processing finished successfully"}
    
    except asyncio.CancelledError:
        logger.warning("Request cancelled by the client.")
        return {"status": "cancelled", "message": "Request was cancelled by the client"}
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        logger.info("Cleaning up resources...")