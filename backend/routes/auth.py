from fastapi import APIRouter, HTTPException, status

from database import users_collection
from schemas.auth import RegisterRequest, LoginRequest, LoginResponse, UserOut
from auth.hashing import hash_password, verify_password
from auth.jwt_handler import create_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest):
    existing = await users_collection.find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_doc = {
        "name": payload.name,
        "email": payload.email,
        "password_hash": hash_password(payload.password),
        "role": payload.role,
    }
    result = await users_collection.insert_one(user_doc)

    return UserOut(
        id=str(result.inserted_id),
        name=payload.name,
        email=payload.email,
        role=payload.role,
    )


@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest):
    user = await users_collection.find_one({"email": payload.email})
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token_data = {
        "sub": str(user["_id"]),
        "email": user["email"],
        "role": user["role"],
        "name": user["name"],
    }
    token = create_access_token(token_data)

    return LoginResponse(
        token=token,
        user=UserOut(
            id=str(user["_id"]),
            name=user["name"],
            email=user["email"],
            role=user["role"],
        ),
    )
