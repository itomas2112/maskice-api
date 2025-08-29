
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.products import router as products_router
from app.api.routes.cart import router as cart_router
from app.api.routes.orders import router as orders_router
from app.api.routes.lookups import router as lookups_router
from app.api.routes.stripe_webhook import router as stripe_router
from app.api.routes.debug import router as debug_router
from app.api.routes.auth import router as auth_router

app = FastAPI(title="Maskino API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", summary="Health")
def root():
    return {"message": "OK"}

# Include routers (paths are identical to original)
app.include_router(products_router)
app.include_router(cart_router)
app.include_router(orders_router)
app.include_router(lookups_router)
app.include_router(stripe_router)
app.include_router(debug_router)
app.include_router(auth_router)